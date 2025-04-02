import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def image_callback(msg):
    # Convert ROS Image message to OpenCV image
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        # Display the image
        cv2.imshow("Image Window", cv_image)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr(f"Failed to convert image: {e}")

def main():
    rospy.init_node('image_subscriber', anonymous=True)
    rospy.Subscriber('/camera/image', Image, image_callback)
    rospy.loginfo("Image subscriber node started. Waiting for images...")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()