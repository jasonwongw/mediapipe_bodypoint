import cv2
import mediapipe as mp
from vpython import *


# mediapipe 模型变量初始化
def mediapipe_varibles_init():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils
    return pose, mp_pose, mp_drawing


# vpython（三维画图）模型变量初始化
def vpython_variables_init():
    points = []
    c = []
    remove=[1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22] #不要的点
    save={} #要的点
    p=0
    for x in range(33):
        if x not in remove:
            points.append(sphere(radius=5, pos=vector(0, -50, 0))) #sphere函数创建一个球 pos(球心的坐标位置)、color(颜色)、radius(半径)、material(材质)  是用来布点的
            save[x]=p #{0: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 23: 7, 24: 8, 25: 9, 26: 10, 27: 11, 28: 12, 29: 13, 30: 14, 31: 15, 32: 16}
            p+=1
        c.append(curve(retain=2, radius=4))  # 曲线retain=2表示保留曲线的两个点用于更新动画，radius=4表示曲线半径为4   是用来连接各个点的。,线不止17个
    #print(points)  #[<vpython.vpython.sphere object at 0x0000025DC9659190>, <vpython.vpython.sphere object at 0x0000025DC9933E50>,... 17
    #print(c)  #[<vpython.vpython.curve object at 0x0000025DC9659670>, <vpython.vpython.curve object at 0x0000025DC98E5B20>, ...
    return points,  save, c


# 在3D里画出骨架的函数
def draw_3d_pose():
    results = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) #得到关节点
    if results.pose_world_landmarks: #存在world关节点
        for i in range(len(save)): #身体
            #点根据骨骼关节点的x，y，z数值改变 ,改变points对象的pos属性(x,y,z)=>改变点的位置
            points[i].pos.x = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark(get_keys(save,i)[0]).value].x * -cap.get(3) #宽
            points[i].pos.y = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark(get_keys(save,i)[0]).value].y * -cap.get(4)
            points[i].pos.z = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark(get_keys(save,i)[0]).value].z * -cap.get(3)

        #画上连接线
        #右躯干
        bodyright=[0,11,23,25,27,29,31,27]
        for i in range(7): #[0,11,23,25,27,29,31,27]
            c[i].append(pos=[vector(points[save[bodyright[i]]].pos.x, points[save[bodyright[i]]].pos.y, points[save[bodyright[i]]].pos.z),
                             vector(points[save[bodyright[i+1]]].pos.x, points[save[bodyright[i+1]]].pos.y, points[save[bodyright[i+1]]].pos.z)])  #vector（）是vpython库中用于创建三维矢量的函数。它采用三个参数来表示向量的x、y和z分量，并返回一个vpython.vector对象

        # 左躯干
        bodyleft = [0, 12, 24, 26, 28, 32, 30, 28]
        for i in range(7):
            c[i+7].append(pos=[
                vector(points[save[bodyleft[i]]].pos.x, points[save[bodyleft[i]]].pos.y, points[save[bodyleft[i]]].pos.z),
                vector(points[save[bodyleft[i + 1]]].pos.x, points[save[bodyleft[i + 1]]].pos.y,
                       points[save[bodyleft[i + 1]]].pos.z)])
        #手臂
        arm = [16, 14, 12, 11, 13, 15]
        for i in range(5):
            c[i + 14].append(pos=[
                vector(points[save[arm[i]]].pos.x, points[save[arm[i]]].pos.y,
                       points[save[arm[i]]].pos.z),
                vector(points[save[arm[i + 1]]].pos.x, points[save[arm[i + 1]]].pos.y,
                       points[save[arm[i + 1]]].pos.z)])

        #腰
        waist = [24,23]
        for i in range(1):
            c[i + 20].append(pos=[
                vector(points[save[waist[i]]].pos.x, points[save[waist[i]]].pos.y,
                       points[save[waist[i]]].pos.z),
                vector(points[save[waist[i + 1]]].pos.x, points[save[waist[i + 1]]].pos.y,
                       points[save[waist[i + 1]]].pos.z)])

    mp_drawing.draw_landmarks(image=f, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)


# 窗口关闭函数
def clos_def():
    cap.release()
    cv2.destroyAllWindows()

def get_keys(d, value):
    return [k for k, v in d.items() if v == value]

# 获取变量
points, save, c = vpython_variables_init()

pose, mp_pose, mp_drawing = mediapipe_varibles_init()
# 打开摄像头，0是第一个摄像头，如果想换一个摄像头请改变这个数字
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('C:/Users/Administrator/Desktop/mediapipe/dance.mp4') #视频
while True:
    # 获取每一帧的图像
    _, f = cap.read()
    # vpython里的一个函数，用来调整3D中的FPS
    rate(100) #动画循环每秒的最大迭代次数设置为150。这意味着动画每秒最多更新150次，即使计算机能够更快地运行代码
    # 调用在3D里画出骨架的函数
    draw_3d_pose()
    # 在每一帧里画骨架
    # 显示每一帧
    cv2.imshow('real_time', f)
    # 检测是否要关闭窗口
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 调用窗口关闭函数
clos_def()