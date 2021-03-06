package grains_lxc.shape;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import grains_lxc.imshow.Imshow;

public class My_Shape {

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	Mat img = null;
	double canny_thresh;
	List<MatOfPoint> contours = null;
	int area_largest_cnt_index = 0;
	Moments moment_largest = null;

	public My_Shape(Mat img_gray) {
		this(img_gray, 20.0);
	}

	public My_Shape(Mat img_gray, double canny_thresh) {
		this.img = img_gray;
		this.canny_thresh = canny_thresh;
		this.find_largest_contours(canny_thresh);
	}

	private void find_largest_contours(double thresh) {
		Mat edges = new Mat();
		contours = new ArrayList<>();
		Imgproc.Canny(this.img, edges, thresh, thresh * 2);
		Imgproc.findContours(edges, this.contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

		double area_largest = 0.0;
		double moment_area = 0.0;

		for (int i = 0; i < contours.size(); i++) {
			Moments moments = Imgproc.moments(contours.get(i));
			if (moments.get_m00() != 0.0) {
				moment_area = moments.get_m00();
				if (area_largest < moment_area) {
					area_largest = moment_area;
					area_largest_cnt_index = i;
					moment_largest = moments;
				}
			}
		}
	}

	public double get_humoments() {
		Mat hu = new Mat();
		Imgproc.HuMoments(this.moment_largest, hu);
		double res = hu.get(0, 0)[0];

		if (res <= 0.0)
			return 0.0;
		if (res >= 1.0)
			return 1.0;
		return res;
	}

	public Mat get_foreground() {
		MatOfPoint2f points = new MatOfPoint2f(this.contours.get(this.area_largest_cnt_index).toArray());
		Point center = new Point();
		float[] radius = new float[1];
		Imgproc.minEnclosingCircle(points, center, radius);

		int square = new Double(2 * radius[0]).intValue();
		int x1 = new Double(center.x - radius[0]).intValue();
		int y1 = new Double(center.y - radius[0]).intValue();

		Rect roi = new Rect(x1, y1, square, square);

		return (new Mat(this.img, roi));
	}

	public void draw_contours_largest() {
		Mat mat = new Mat(this.img.rows(), this.img.cols(), CvType.CV_8UC1);
		mat.setTo(new Scalar(0));

		Imgproc.drawContours(mat, this.contours, this.area_largest_cnt_index, new Scalar(255));

		Imshow ims2 = new Imshow("largest");
		ims2.showImage(mat);
	}
}