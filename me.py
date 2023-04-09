import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

mp_drawing = mp.solutions.drawing_utils #绘制标记点或边界框(py文件)
mp_drawing_styles = mp.solutions.drawing_styles #提供了一些常用的绘图样式和颜色(py文件)
mp_pose = mp.solutions.pose #检测和估计人体姿态的方法。这个模块可以接收图像或视频数据作为输入，并输出包含人体关键点位置信息、骨架连接信息、身体朝向角度等信息的结果。它可以用于许多应用程序，例如运动分析、姿势校正、互动体感游戏等。(py文件)

# For webcam input:
#cap = cv2.VideoCapture(0)  #摄像头
video_cap = cv2.VideoCapture('C:/Users/Administrator/Desktop/mediapipe/dance.mp4') #视频
# Get some video parameters to generate output video with classificaiton.
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT) #视频总的帧数
video_fps = video_cap.get(cv2.CAP_PROP_FPS) ## 获取视频帧速率 FPS
# 获取视频帧宽度和高度
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Open output video.
out_video = cv2.VideoWriter('dance_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))
'''
static_image_mode=False,  如果为True，则假定输入为静态图像，并将启用相应的优化。默认为False，即假定输入为视频流或连续帧
model_complexity=1, 模型复杂度，可以设置为0、1或2，分别代表高、中和低三种不同的复杂度级别。默认为1，即中等复杂度
smooth_landmarks=True,一个布尔值，表示是否在输出的关键点上启用平滑滤波器。默认为True。
enable_segmentation=False,一个布尔值，表示是否启用背景分割功能。默认为False
smooth_segmentation=True,一个布尔值，表示是否在输出的分割掩码上启用平滑滤波器。默认为True。
min_detection_confidence=0.5,一个浮点数，表示进行姿态检测时所需的最小置信度得分。默认为0.5。
min_tracking_confidence=0.5 一个浮点数，表示进行姿态跟踪时所需的最小置信度得分。默认为0.5。
'''
with mp_pose.Pose(  #Pose类
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while True:#cap.isOpened():
    success, image = video_cap.read()#cap.read()
    if not success:
      #print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break #continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False #数组是否可写入（即是否可以修改数组的值）。如果 image.flags.writeable 为False，则数组不可写
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #opencv的BGR->RGB
    results = pose.process(image)  #处理图片返回结果，包括results.pose_landmarks.landmark(Landmark（地标）对象，坐标点xyz和可信度)和results.pose_world_landmarks.landmark
    #print(mp_pose.PoseLandmark(5) ) #PoseLandmark.RIGHT_EYE  .value=5
    ''' 
        results.pose_landmarks.landmark(33个点) 相对于图像或视频中人物的二维关键点（landmark）的坐标值,提供的是在图像或视频中的相对位置信息，
        x: 0.4883374273777008
        y: 2.814537763595581
        z: 0.1792668253183365
        visibility: 0.00016459620383102447
        
        results.pose_world_landmarks.landmark(33个点)  包含的是相对于世界坐标系的三维关键点的坐标值,以米为单位的真实世界3D坐标，原点位于臀部之间的中心。提供的是真实世界中的绝对位置信息。
        x: -0.008830961771309376
        y: 0.014062387868762016
        z: 0.030819499865174294
        visibility: 5.5312590120593086e-05
    '''
    # Draw the pose annotation on the image.
    image.flags.writeable = True #数组是否可写入（即是否可以修改数组的值）。如果 image.flags.writeable 为True，则数组可写

    # 画坐标点并连线
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks, #点
        mp_pose.POSE_CONNECTIONS, #点连接，画线
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())  #颜色

    # 在三维真实物理坐标系中可视化   米为单位
    img = mp_drawing.plot_landmarks(results.pose_world_landmarks,
                                    mp_pose.POSE_CONNECTIONS)# ,w=image.shape[0],h=image.shape[1])
    # ①分别展示，两个imshow
    # if img is not None:
    #     cv2.imshow('3D Pose', img)
    # ②3d pose嵌入在image中
    if img is not None:
        image = Image.fromarray(image) # #将 Numpy 数组转换成 PIL 图像对象以进行后续操作,  (cv2转变为PIL进行图像显示)  W,H,C
        img.thumbnail((int(image.size[0] * 0.4),  #numpy是shape,图像是size
                       int(image.size[1] * 0.4)),
                      Image.ANTIALIAS)  # 裁剪图片(注意，函数参数是一个(x,y)尺寸的元组)，用于调整图像大小，使其缩放后保持更高的质量，减少锯齿和失真等不良效果
        image.paste(img, (int(image.size[0] * 0.05), int(image.size[1] * 0.05)))  # box：指定粘贴位置的矩形框，由左上角和右下角坐标组成的元组或列表

    #显示图像(cv2格式，bgr) ①
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1)) #镜像
    #②PIL图像->cv2
    # img_PIL = np.array(image)  # 先转换为数组   H W C
    # image = cv2.cvtColor(img_PIL, cv2.COLOR_RGB2BGR)
    # cv2.imshow('MediaPipe Pose', image)  # 镜像
    #保存图像视频
    out_video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    # if cv2.waitKey(5) & 0xFF == 27:
    #   break
video_cap.release()#cap.release()