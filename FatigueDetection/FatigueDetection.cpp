#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <typeinfo>
using namespace cv;
using namespace std;


struct faceDetection
{
	Mat srcImage;
	Mat faceRoi;
	float coor_x;
	float coor_y;
	bool FLAG;

};
struct eyeDetection
{
	Mat faceImage;
	Mat eyeRoi;
	bool FLAG; //该标志位代表是否检测到眼睛，FLAG=1表示检测到，然后进行椭圆拟合；如果FLAG=0则不进行椭圆拟合
};

struct EllipseEye
{
	Mat res;
	float width;
	float height;
	bool FLAG;
};

Mat resShowImage;
Mat eyeROI;
bool eyeFLAG;
Mat faceRectImage;
Mat faceROI;
bool faceFLAG;
float coor_x;
float coor_y;
bool flag;
double scale = 1;
VideoCapture cap(0);
int nHeadPosture = 0;
float PERCLOS;
ofstream outFile;
//float eyeroi_height;
//float eyeroi_width;
//float eyeroi_center[2];
//float eyeroi_area;
bool ellipseFLAG;

CascadeClassifier cascadeFace;
CascadeClassifier cascadeEye;
CascadeClassifier cascadeEyeWithGlasses;

const string cascadeNameFace = "haarcascade_frontalface_alt2.xml";
const string cascadeNameEyeWithGlasses = "haarcascade_eye_tree_eyeglasses.xml";
const string cascadeNameEyeNoGlasses = "haarcascade_eye.xml";

void process(Mat image, bool flag);
struct faceDetection facedetection(Mat image, CascadeClassifier cascade);
struct eyeDetection eyedetection(Mat image, CascadeClassifier cascade, Mat faceRectImage, float coor_x, float coor_y);
struct EllipseEye eyeEllipse(Mat image);


int main()
{
	cascadeFace.load(cascadeNameFace);
	cascadeEye.load(cascadeNameEyeNoGlasses);
	cascadeEyeWithGlasses.load(cascadeNameEyeWithGlasses);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	clock_t start, end;
	Mat frame;	
	outFile.open("data.csv", ios::out);
	outFile << "width" << "," << "height" << "," << "rate" << "," << "headpos" << endl;   //headpos=1表示为头部姿态不正，headpos=0表示头部姿态正
	while (1)
	{
		start = clock();
		cap >> frame;
		process(frame, false);
		end = clock();
		cout << "SPEED: " << end - start << endl;
		if (waitKey(10) == 27)        // 按下esc结束程序
			break;
	}
	outFile.close();
	return 0;
}

void process(Mat image, bool flag)
{
	// 人脸检测...
	struct faceDetection face = facedetection(image, cascadeFace);
	faceFLAG = face.FLAG;
	faceRectImage = face.srcImage;
	faceROI = face.faceRoi;
	coor_x = face.coor_x;
	coor_y = face.coor_y;
	cout << "FACE DETECTION DONE..." << endl;
	if (faceFLAG)                               // 检测到人脸，继续下面步骤
	{
		// 眼睛检测...
		if (flag)                              // 不戴眼镜
		{
			struct eyeDetection eye = eyedetection(faceROI, cascadeEye, faceRectImage, coor_x, coor_y);
			resShowImage = eye.faceImage;        // for Show
			eyeROI = eye.eyeRoi;    // for NEXT PROCESS
			eyeFLAG = eye.FLAG;			
		}
		else                                   // 戴眼镜
		{
			struct eyeDetection eye = eyedetection(faceROI, cascadeEyeWithGlasses, faceRectImage, coor_x, coor_y);
			resShowImage = eye.faceImage;         // for Show
			eyeROI = eye.eyeRoi;   // for NEXT PROCESS
			eyeFLAG = eye.FLAG;;
		}
		imshow("检测结果", resShowImage);
		cout << "EYE DETECTION DONE..." << endl;
		
		if (eyeFLAG)          // 检测到眼睛，继续下一步骤
		{
			resize(eyeROI, eyeROI, Size(24, 24));
			struct EllipseEye e = eyeEllipse(eyeROI);
			if (e.FLAG)
			{
				resize(e.res, e.res, Size(e.res.rows*10, e.res.cols*10));
				imshow("椭圆拟合", e.res);
				outFile << e.width << "," << e.height << "," << e.width / e.height << "," << 0 << endl;
				cout << "ELLIPSEFIT DONE..." << endl;
			}
			else 
			{
				outFile << 0 << "," << 0 << "," << 0 << "," << 0 << endl;
			}
		}
		else         // 未检测到眼睛，则不进行椭圆拟合，判定为闭眼状态，PERCLOS = 0；
		{
			outFile << 0 << "," << 0 << "," << 0 << "," << 0 << endl;
			//PERCLOS = 0;
		}
	}
	else  // 未检测到人脸，则不进行下面步骤，判断头部姿态不正
	{
		outFile << 0 << "," << 0 << "," << 0 << "," << 1 << endl;
		nHeadPosture++;
	}	
}

struct faceDetection facedetection(Mat image, CascadeClassifier cascade)
{
	struct faceDetection face;
	Mat tmp;
	image.copyTo(tmp);
	vector<Rect> faces;
	Mat grayImage;
	cvtColor(tmp, grayImage, CV_BGR2GRAY);

	cascade.detectMultiScale(grayImage, faces, 1.1, 2, CASCADE_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(40, 40));   // Size()可以决定检测速度，但是size越大侧方向检测效果越不好

	int rect_x, rect_y, rect_width, rect_height;

	if (faces.size() > 0)
	{
		for (int i = 0; i < faces.size(); i++)
		{
			rect_x = faces[0].x;             //  这里！！！ 可以通过设置只选择一个疑似脸部区域  但是！！！有可能错误检测
			rect_y = faces[0].y;
			rect_width = faces[0].width;
			rect_height = faces[0].height;
			rectangle(tmp, Point(rect_x, rect_y), Point(rect_x + rect_width, rect_y + rect_height), Scalar(0, 0, 255), 1);
			face.faceRoi = tmp(Rect(rect_x, rect_y, rect_width, rect_height));
			face.coor_x = rect_x;
			face.coor_y = rect_y;
		}
		face.srcImage = tmp;
		face.FLAG = true;
		return face;
	}
	else
	{
		face.srcImage = tmp;
		face.FLAG = false;
		return face;
	}
	
}

struct eyeDetection eyedetection(Mat image, CascadeClassifier cascade, Mat faceRectImage, float coor_x, float coor_y)
{
	struct eyeDetection eye;
	Mat tmp;
	image.copyTo(tmp);
	vector<Rect> eyes;
	Mat grayImage;
	cvtColor(tmp, grayImage, CV_BGR2GRAY);

	cascade.detectMultiScale(grayImage, eyes, 1.1, 2, CASCADE_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(20, 20));   // Size()可以决定检测速度，但是size越大侧方向检测效果越不好

	int rect_x, rect_y, rect_width, rect_height;
	if (eyes.size() > 0)
	{
		for (int i = 0; i < eyes.size(); i++)
		{
			rect_x = eyes[0].x;   // 这里！！！ 可以通过设置选择某一只眼睛  但是！！！有可能错误检测
			rect_y = eyes[0].y;
			rect_width = eyes[0].width;
			rect_height = eyes[0].height;
			rectangle(faceRectImage, Point(rect_x + coor_x, rect_y + coor_y), Point(rect_x + coor_x + rect_width, rect_y + coor_y + rect_height), Scalar(0, 0, 255), 1);  // 这里！在原始图像上标注出来眼睛框！！！
			eye.eyeRoi = tmp(Rect(rect_x, rect_y, rect_width, rect_height));
		}
		eye.faceImage = faceRectImage;
		eye.FLAG = true;
		return eye;
	}
	else
	{
		eye.faceImage = faceRectImage;
		eye.FLAG = false;
		return eye;
	}
	
}


struct EllipseEye eyeEllipse(Mat image)
{
	Mat tmp;
	image.copyTo(tmp);
	//eyeroi_height = tmp.cols;      // 为椭圆拟合提供参照，选择center最接近eyeROI中心的椭圆作为瞳孔
	//eyeroi_width = tmp.rows;
	//eyeroi_center[0] = eyeroi_height / 2;
	//eyeroi_center[1] = eyeroi_width / 2;
	//eyeroi_area = eyeroi_height*eyeroi_width;

	struct EllipseEye e;
	Mat hsv;
	cvtColor(tmp, hsv, CV_BGR2HSV);

	vector<Mat>m;
	split(hsv, m);
	vector<Mat> H_channel;
	split(hsv, H_channel);
	H_channel[1] = 0;
	H_channel[2] = 0;
	merge(H_channel, m[0]);

	Mat H_img = m[0];
	Mat gray;
	cvtColor(H_img, gray, CV_BGR2GRAY);
	Mat dst;
	threshold(gray, dst, 0, 255, CV_THRESH_OTSU);

	dst = 255 - dst;

	Mat ele = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(dst, dst, ele);
	
	//imwrite("dst.jpg", dst);
	vector<vector<Point>> contours;
	findContours(dst, contours, RETR_LIST, CHAIN_APPROX_NONE);

	for (size_t i = 0; i < contours.size(); i++)
	{
		size_t count = contours[i].size();
		if (count < 5)   // 至少5个点才能进行拟合
			continue;

		Mat pointsf;
		Mat(contours[i]).convertTo(pointsf, CV_32F);
		
		RotatedRect box = fitEllipse(pointsf);
		
		//cout << box.size.area() << endl;

		if (box.size.area() >= 192 || box.size.height == box.size.width)       // 三分之一
		{
			continue;
		}
	
		if (!box.size.empty())
		{
			e.width = box.size.width;
			e.height = box.size.height;
			cout << ">>>>WIDTH: " << e.width << endl;
			cout << ">>>>HEIGHT: " << e.height << endl;

			ellipse(tmp, box, Scalar(0, 0, 255), 1, CV_AA);
			e.res = tmp;
			e.FLAG = true;
			//imwrite("test.jpg", e.res);
			return e;
		}
		else
		{
			e.FLAG = false;
			return e;
		}			
	}

	e.FLAG = false;           // 这里！！！ 防止所有box都不满足条件的时候，跳出循环并且判定FLAG=false！ 很牛逼！
	return e;
}


