

// for std
#include <iostream>
// opencv
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "matplotlibcpp.h"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>

#include <boost/concept_check.hpp>
// g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include "g2o/types/sba/types_six_dof_expmap.h"
#include <g2o/types/sba/edge_sba_scale.h>
#include <g2o/stuff/sampler.h>
#include "g2o/core/sparse_optimizer_terminate_action.h"
#include <cmath>
using namespace std;
using namespace cv;
using namespace g2o;
//
// find corresponding points in two imgs
// input : img1, img2
// output: points1, points2, 2 sets of corresponding points
int findCorrespondingPoints(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Point2f> &points1, vector<cv::Point2f> &points2);

std::tuple<double, double> returnTheRotationAngle(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Point2f> &pts1, vector<cv::Point2f> &pts2,
Eigen::Quaterniond first_frame_sensor,Eigen::Quaterniond second_frame_sensor,const double remove_percent,g2o::SE3Quat temp1111);


g2o::SE3Quat returnThenewpose(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Point2f> &pts1, vector<cv::Point2f> &pts2,
Eigen::Quaterniond first_frame_sensor,Eigen::Quaterniond second_frame_sensor, const double remove_percent, g2o::SE3Quat Pose_lastloop);

// camera intrinsic parameter

//1920 1080
double cx = 929.5655209;
double cy = 606.73290867;
double fx = 869.28853317;
double fy = 867.95463952;
double k1 = -0.45517004;
double k2 = 0.18440269;
double k3 = 0.17406266;
double p1 = -0.00273677;
double p2 = 0.0091266;


namespace plt = matplotlibcpp;


class G2O_TYPES_SBA_API my_edge : public BaseBinaryEdge<2, Vector2, VertexPointXYZ, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    my_edge() : BaseBinaryEdge<2, Vector2, VertexPointXYZ, VertexSE3Expmap>()
    {
        _cam = 0;
        resizeParameters(1);
        installParameter(_cam, 0);
    }
    bool read(std::istream &is)
    {
        readParamIds(is);
        g2o::internal::readVector(is, _measurement);
        return readInformationMatrix(is);
    }

    bool write(std::ostream &os) const
    {
        writeParamIds(os);
        g2o::internal::writeVector(os, measurement());
        return writeInformationMatrix(os);
    }
    void computeError()
    {
        const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[1]);
        const VertexPointXYZ *v2 = static_cast<const VertexPointXYZ *>(_vertices[0]);
        const CameraParameters *cam =
            static_cast<const CameraParameters *>(parameter(0));

        _error = measurement() - cam->cam_map(v1->estimate().map(v2->estimate()));
    }

    /*
        todo change something about the jacobian/error
    */
    void linearizeOplus()
    {
        VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
        SE3Quat T(vj->estimate());
        VertexPointXYZ *vi = static_cast<VertexPointXYZ *>(_vertices[0]);
        Vector3 xyz = vi->estimate();
        Vector3 xyz_trans = T.map(xyz);

        number_t x = xyz_trans[0];
        number_t y = xyz_trans[1];
        number_t z = xyz_trans[2];
        number_t z_2 = z * z;

        const CameraParameters *cam =
            static_cast<const CameraParameters *>(parameter(0));

        Eigen::Matrix<number_t, 2, 3, Eigen::ColMajor> tmp;
        tmp(0, 0) = cam->focal_length;
        tmp(0, 1) = 0;
        tmp(0, 2) = -x / z * cam->focal_length;

        tmp(1, 0) = 0;
        tmp(1, 1) = cam->focal_length;
        tmp(1, 2) = -y / z * cam->focal_length;

        _jacobianOplusXi = -1. / z * tmp * T.rotation().toRotationMatrix();

        // page 253 
        // rotation
        _jacobianOplusXj(0, 0) = x * y / z_2 * cam->focal_length;
        _jacobianOplusXj(0, 1) = -(1 + (x * x / z_2)) * cam->focal_length;
        _jacobianOplusXj(0, 2) = y / z * cam->focal_length;
        _jacobianOplusXj(1, 0) = (1 + y * y / z_2) * cam->focal_length;
        _jacobianOplusXj(1, 1) = -x * y / z_2 * cam->focal_length;
        _jacobianOplusXj(1, 2) = -x / z * cam->focal_length;
        // translation
        _jacobianOplusXj(0, 3) = -1. / z * cam->focal_length;
        _jacobianOplusXj(0, 4) = 0;
        _jacobianOplusXj(0, 5) = x / z_2 * cam->focal_length;
        _jacobianOplusXj(1, 3) = 0;
        _jacobianOplusXj(1, 4) = -1. / z * cam->focal_length;
        _jacobianOplusXj(1, 5) = y / z_2 * cam->focal_length;
    }

public:
    CameraParameters *_cam; // TODO make protected member?
};



class G2O_TYPES_SBA_API camera_edge : public BaseBinaryEdge<1, number_t, VertexSE3Expmap, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    camera_edge() : BaseBinaryEdge<1, number_t, VertexSE3Expmap, VertexSE3Expmap>() {}

    bool read(std::istream &is)
    {
        number_t meas;
        is >> meas;
        setMeasurement(meas);
        information().setIdentity();
        is >> information()(0, 0);
        return true;
    }

    bool write(std::ostream &os) const
    {
        os << measurement() << " " << information()(0, 0);
        return os.good();
    }

    void computeError()
    {

        // camera vertex v1 and v2
        VertexSE3Expmap *v1 = dynamic_cast<VertexSE3Expmap *>(_vertices[0]);
        VertexSE3Expmap *v2 = dynamic_cast<VertexSE3Expmap *>(_vertices[1]);
        
        // rotation matrix of v1_r and v2_r 
        // T * V1 = V2
        // T = V1 * V2^-1
        g2o::Quaternion v1_rotation = v1->estimate().rotation();
        g2o::Quaternion v2_rotation = v2->estimate().rotation();
        Eigen::Matrix3d v1_rotation_matix = v1_rotation.normalized().toRotationMatrix();
        Eigen::Matrix3d v2_rotation_matix = v2_rotation.normalized().toRotationMatrix();
        Eigen::Matrix3d temp = v1_rotation_matix * v2_rotation_matix.inverse();
        AngleAxis axis_temp(temp);
        double error_ = _measurement - axis_temp.angle();



        _error[0] = error_;
    }
    
};



class G2O_TYPES_SBA_API rcm_constraints : public BaseBinaryEdge<1, number_t, VertexSE3Expmap, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    rcm_constraints() : BaseBinaryEdge<1, number_t, VertexSE3Expmap, VertexSE3Expmap>() {}

    bool read(std::istream &is)
    {
        number_t meas;
        is >> meas;
        setMeasurement(meas);
        information().setIdentity();
        is >> information()(0, 0);
        return true;
    }

    bool write(std::ostream &os) const
    {
        os << measurement() << " " << information()(0, 0);
        return os.good();
    }

    void computeError()
    {

        // camera vertex v1 and v2
        VertexSE3Expmap *v1 = dynamic_cast<VertexSE3Expmap *>(_vertices[0]);
        VertexSE3Expmap *v2 = dynamic_cast<VertexSE3Expmap *>(_vertices[1]);
        
        // r_23*t1 - r_12*t2 = 0
  
        g2o::SE3Quat v1_pose(v1->estimate());
        g2o::SE3Quat v2_pose(v2->estimate());
        g2o::SE3Quat T = v1_pose * v2_pose.inverse();
        g2o::Matrix3 r_matrix = T.rotation().toRotationMatrix();
        double temp = r_matrix(1,2) * T.translation()(0) - r_matrix(0,1) *T.translation()(1);
        
        double error_ = _measurement - temp;
        _error[0] = error_;
    }
    
};

g2o::SparseOptimizer optimizer;

int flag = 0;

int main(int argc, char **argv)
{   
    
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver(new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>());
    std::unique_ptr<g2o::BlockSolver_6_3> block_solver(new g2o::BlockSolver_6_3(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg *algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(false);
    g2o::SparseOptimizerTerminateAction* terminateAction = new g2o::SparseOptimizerTerminateAction;
    optimizer.addPostIterationAction(terminateAction);


    // read imgs
    int name = 20157; 
    string address = "./../../../1920_120fps/";
    int n = 32;
    std::vector<double> our_e(n), sensor_e(n);
    // start from 20157
    std::vector<int> frames = {20157,20265,20373,20481,20589,20697,20805,20913,21021,21129,21237,21345,21453,21561,21669,21777,21885,21993,
    22101,22209,22317,22425,22533,22641,22749,22857,22965,23073,23181,23289,23397};
    // in order, data from file
    std::vector<double> w = {-0.891969,-0.892961,-0.893355,-0.89409,-0.894854,-0.895011,-0.895301,-0.895311,-0.895517,-0.895935,-0.894547,
    -0.894636,-0.895503,-0.896076,-0.895765,-0.896675,-0.897269,-0.89703,-0.897395,-0.89796,-0.896948,-0.896873,-0.896515,-0.895647,-0.894901,-0.895336,
    -0.89484,-0.894176,-0.894659,-0.894021,-0.893924,-0.893977};
    std::vector<double> x = {0.190845,0.188844,0.184472,0.18593,0.188577,0.189417,0.189232,0.190398,0.188912,0.182014,0.174681,0.17078,0.167659,0.163344,
    0.157115,0.153698,0.146726,0.141163,0.139128,0.13201,0.122183,0.115386,0.11149,0.103717,0.097617,0.090643,0.085394,0.07773,0.08065,0.073867,0.075334,
    0.080212,0.082838};
    std::vector<double> y = {0.352847,0.352152,0.356372, 0.356791,0.357861,0.359336,0.360328,0.36067,0.362585,0.363812,0.370127,0.372659,0.376769,0.378741,
    0.378552,0.380276,0.380146,0.379642,0.380963,0.380967,0.381779,0.385781,0.386655,0.388957,0.391574,0.394125,0.393937,0.395668,0.398014,0.397412,0.401701,
    0.405257,0.407127};
    std::vector<double> z = {-0.208492,-0.207237,-0.202204,-0.196807,-0.188706,-0.184265,-0.181082,-0.179118,-0.175765,-0.17836,-0.179663,-0.177723,-0.168837,
    -0.167176,-0.170473,-0.171381,-0.173016,-0.175651,-0.175636,-0.17922,-0.181615,-0.1826,-0.18354,-0.184978,-0.186968,-0.188682,-0.189462,-0.191482,-0.188492,
    -0.190246,-0.183541,-0.173877,-0.167902};
    
    g2o::SE3Quat Pose_lastloop;

    for(int i=0; i<n;i++)
    {   
        g2o::VertexSE3Expmap *estimation1111 = new g2o::VertexSE3Expmap();
        string img1_name = address + std::to_string(name) + ".png";
        cout << "first name : "<<  img1_name <<endl;
        name += 30;
        string img2_name = address + std::to_string(name) + ".png";
        cout << "second name : "<< img2_name <<endl;
        cout << "i: " << i << endl;        
        cv::Mat img1 = cv::imread(img1_name);
        cv::Mat img2 = cv::imread(img2_name);

        Eigen::Quaterniond first_frame(w[i],x[i],y[i],z[i]);
        Eigen::Quaterniond second_frame(w[i+1],x[i+1],y[i+1],z[i+1]);
        vector<cv::Point2f> pts1, pts2;
        g2o::SE3Quat tempoutput;
        tempoutput = returnThenewpose(img1,img2,pts1,pts2,first_frame,second_frame, 0, Pose_lastloop);
        
        cout << "flag: "<<flag << endl;
        Pose_lastloop.setRotation(tempoutput.rotation());
        Pose_lastloop.setTranslation(tempoutput.translation());
        
        
        
    }
    optimizer.save("Output.g2o");

   

    return 0;
}

int findCorrespondingPoints(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Point2f> &points1, vector<cv::Point2f> &points2)
{
    // SURF
    int minHessian = 400;
    Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
    detector->setHessianThreshold(minHessian);

    // sift
    // Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();

    // orb
    // Ptr<cv::ORB> detector =  cv::ORB::create();

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desp1, desp2;
    detector->detectAndCompute(img1, noArray(), kp1, desp1);
    detector->detectAndCompute(img2, noArray(), kp2, desp2);

    cout << "found " << kp1.size() << " and " << kp2.size() << " feature points respectively" << endl;

    // DescriptorMatcher::FLANNBASED
    //"BruteForce-Hamming"

    /*
    change mathcer and descriptor 
    */

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming");
    // cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();
    // cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(NORM_HAMMING);

    // ++
    double knn_match_ratio = 0.8;
    // matches_flann
    vector<vector<cv::DMatch>> matches_knn;
    // knn
    matcher->knnMatch(desp1, desp2, matches_knn, 2);
    vector<cv::DMatch> matches;
    for (size_t i = 0; i < matches_knn.size(); i++)
    {
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance)
            matches.push_back(matches_knn[i][0]);
    }
    cout << "found " << matches.size() << " matching points" << endl;

    for (auto m : matches)
    {
        points1.push_back(kp1[m.queryIdx].pt);
        points2.push_back(kp2[m.trainIdx].pt);
    }


    
    
    return true;
}

g2o::SE3Quat returnThenewpose(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Point2f> &pts1, vector<cv::Point2f> &pts2,
Eigen::Quaterniond first_frame_sensor,Eigen::Quaterniond second_frame_sensor, const double remove_percent, g2o::SE3Quat Pose_lastloop)
{
   
    
    if (findCorrespondingPoints(img1, img2, pts1, pts2) == false)
    {
        cout << "Not enough matching points" << endl;
        return  g2o::SE3Quat();
    }
    cout << "found " << pts1.size() << " sets of matching points" << endl;

    // remove_percent, always 0, leave for further development
    if (remove_percent != 0)
    {
        std::vector<int> removeindex;
        for(int i=0; i<pts1.size(); i++)
        {       
            removeindex.push_back(i);
        }
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(removeindex.begin(), removeindex.end(), std::default_random_engine(seed));
        
        int remove_size = pts1.size() * remove_percent;
        cout << " !!remove " << remove_size << " sets of matching points" << endl;
        for (int j = 0; j<remove_size; j++)
        {
            pts1.erase(pts1.begin()+removeindex[j]);
            pts2.erase(pts2.begin()+removeindex[j]); 
        } 
             
        
        cout << "found " << pts1.size() << " sets of matching points" << endl;    
    }
  
    g2o::VertexSE3Expmap *first_camera = new g2o::VertexSE3Expmap();
    first_camera->setId(flag+0);
    g2o::SE3Quat frame_one_measure;
    first_camera->setEstimate(Pose_lastloop);
    first_camera->setFixed(true);
    optimizer.addVertex(first_camera);

    
    g2o::VertexSE3Expmap *second_camera = new g2o::VertexSE3Expmap();
    second_camera->setId(flag+1);
    g2o::SE3Quat frame_two_measure;
 
    second_camera->setEstimate(Pose_lastloop);
    optimizer.addVertex(second_camera);

    // many feature points vertexs
    // first img as the initial
    // pts1 : keypoints in the first img
    // this are all feature points
    for (size_t i = 0; i < pts1.size(); i++)
    {

        g2o::VertexPointXYZ *v = new g2o::VertexPointXYZ();
        // first 2 are camera poses
        v->setId(flag+2 + i);
        // we dont know the depth, set as 1
        double z = 1;
        double r = sqrt( pow(pts1[i].x,2) +  pow(pts1[i].y,2));
        double dist_ratio = 1 + k1 * pow(r,2) + k2 * pow(r,4) + k3 * pow(r,6);
        double x = (pts1[i].x - cx) * z / fx / dist_ratio;
        double y = (pts1[i].y - cy) * z / fy / dist_ratio;

        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(x, y, z));
        optimizer.addVertex(v);
    }

    // prepare the camera para
    g2o::CameraParameters *camera = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
    camera->setId(flag+0);
    optimizer.addParameter(camera);

    // reprojection error are saved in edges
    // in first img
    vector<my_edge *> edges;
    // vector<g2o::EdgeProjectXYZ2UV *> edges;
    for (size_t i = 0; i < pts1.size(); i++)
    {
        my_edge *edge = new my_edge();
        // g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(flag+i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(flag+0)));
        edge->setMeasurement(Eigen::Vector2d(pts1[i].x, pts1[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        // kernel function to against noise
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }
    // second img
    for (size_t i = 0; i < pts2.size(); i++)
    {
        my_edge *edge = new my_edge();
        // g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(flag+i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(flag+1)));
        edge->setMeasurement(Eigen::Vector2d(pts2[i].x, pts2[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    // Measurement constraint
    camera_edge *extra_edge = new camera_edge();
    extra_edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0)));
    extra_edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1)));
    Eigen::Matrix3d first_frame_sensor_matix = first_frame_sensor.normalized().toRotationMatrix();
    Eigen::Matrix3d second_frame_sensor_matix = second_frame_sensor.normalized().toRotationMatrix();
    Eigen::Matrix3d temp2 = first_frame_sensor_matix * second_frame_sensor_matix.inverse();
    AngleAxis axis_temp2(temp2);
    double relative_extimation = axis_temp2.angle();
    extra_edge->setMeasurement(relative_extimation);
    Eigen::Matrix<double, 1, 1> info(pts1.size());
    cout << "checksssss" << info << endl; 
    extra_edge->setInformation(info);
    extra_edge->setParameterId(0, 0);
    extra_edge->setRobustKernel(new g2o::RobustKernelHuber());
    optimizer.addEdge(extra_edge);
   
    // RCM constraint
    rcm_constraints *rcm_edge = new rcm_constraints();
    rcm_edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(flag+0)));
    rcm_edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(flag+1)));
    rcm_edge->setMeasurement(0);
    Eigen::Matrix<double, 1, 1> rcm_info(pts1.size());
    rcm_edge->setInformation(rcm_info);
    rcm_edge->setParameterId(0, 0);
    rcm_edge->setRobustKernel(new g2o::RobustKernelHuber());
    optimizer.addEdge(rcm_edge);


    cout << "start to optimize" << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(15);
    
    cout << "optimize done" << endl;
    g2o::VertexSE3Expmap *result = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(flag+1));
    g2o::SE3Quat output = result->estimate();
    flag += 2 + pts1.size();

    return output;
   
}



