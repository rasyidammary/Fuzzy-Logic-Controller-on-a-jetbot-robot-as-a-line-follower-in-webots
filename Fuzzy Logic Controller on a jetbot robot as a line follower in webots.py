import skfuzzy as fuzz
from skfuzzy import control as ctrl
from controller import Robot, Camera
import cv2
import numpy as np

# Inisialisasi robot dan kamera
TIME_STEP = 64
robot = Robot()
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

# Inisialisasi variabel linguistik dan sistem kendali fuzzy
angle = ctrl.Antecedent(np.arange(-90, 91, 1), 'angle')
velocity = ctrl.Antecedent(np.arange(0, 256, 1), 'velocity')
left_speed = ctrl.Consequent(np.arange(0, 21, 1), 'left_speed')
right_speed = ctrl.Consequent(np.arange(0, 21, 1), 'right_speed')

# Membuat himpunan linguistik untuk variabel angle
angle['kiri'] = fuzz.trimf(angle.universe, [-90, -90, 0])
angle['tengah'] = fuzz.trimf(angle.universe, [-10, 0, 10])
angle['kanan'] = fuzz.trimf(angle.universe, [0, 90, 90])

# Membuat himpunan linguistik untuk variabel velocity
velocity['lambat'] = fuzz.trimf(velocity.universe, [0, 0, 100])
velocity['sedang'] = fuzz.trimf(velocity.universe, [50, 127, 200])
velocity['cepat'] = fuzz.trimf(velocity.universe, [150, 255, 255])

# Membuat himpunan linguistik untuk variabel left_speed dan right_speed
left_speed['berhenti'] = fuzz.trimf(left_speed.universe, [0, 0, 0])
left_speed['lambat'] = fuzz.trimf(left_speed.universe, [0, 5, 10])
left_speed['sedang'] = fuzz.trimf(left_speed.universe, [5, 10, 15])
left_speed['cepat'] = fuzz.trimf(left_speed.universe, [10, 20, 20])

right_speed['berhenti'] = fuzz.trimf(right_speed.universe, [0, 0, 0])
right_speed['lambat'] = fuzz.trimf(right_speed.universe, [0, 5, 10])
right_speed['sedang'] = fuzz.trimf(right_speed.universe, [5, 10, 15])
right_speed['cepat'] = fuzz.trimf(right_speed.universe, [10, 20, 20])

# Membuat aturan kendali fuzzy
rule1 = ctrl.Rule(velocity['lambat'] & angle['kiri'], (left_speed['berhenti'], right_speed['cepat']))
rule2 = ctrl.Rule(velocity['lambat'] & angle['tengah'], (left_speed['lambat'], right_speed['lambat']))
rule3 = ctrl.Rule(velocity['lambat'] & angle['kanan'], (left_speed['cepat'], right_speed['berhenti']))
rule4 = ctrl.Rule(velocity['sedang'] & angle['kiri'], (left_speed['lambat'], right_speed['cepat']))
rule5 = ctrl.Rule(velocity['sedang'] & angle['tengah'], (left_speed['sedang'], right_speed['sedang']))
rule6 = ctrl.Rule(velocity['sedang'] & angle['kanan'], (left_speed['lambat'], right_speed['cepat']))
rule7 = ctrl.Rule(velocity['cepat'] & angle['kiri'], (left_speed['sedang'], right_speed['cepat']))
rule8 = ctrl.Rule(velocity['cepat'] & angle['tengah'], (left_speed['cepat'], right_speed['cepat']))
rule9 = ctrl.Rule(velocity['cepat'] & angle['kanan'], (left_speed['cepat'], right_speed['sedang']))
rule10 = ctrl.Rule(angle['tengah'], (left_speed['sedang'], right_speed['sedang']))

# Membuat sistem kendali fuzzy
speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
robot_ctrl = ctrl.ControlSystemSimulation(speed_ctrl)

while robot.step(TIME_STEP) != -1:
    img = camera.getImageArray()
    img = np.array(img, dtype=np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY)
    
    h, w = img_binary.shape
    boundary1 = int(h / 3)
    boundary2 = int(2 * h / 3)
    
    image1 = img_binary[0:boundary1, 0:int(w / 3)]
    image2 = img_binary[0:boundary1, int(w / 3):int(2 * w / 3)]
    image3 = img_binary[0:boundary1, int(2 * w / 3):w]
    image4 = img_binary[boundary1:boundary2, 0:int(w / 3)]
    image5 = img_binary[boundary1:boundary2, int(w / 3):int(2 * w / 3)]
    image6 = img_binary[boundary1:boundary2, int(2 * w / 3):w]
    
    sum1 = np.sum(image1) / 255
    sum2 = np.sum(image2) / 255
    sum3 = np.sum(image3) / 255
    sum4 = np.sum(image4) / 255
    sum5 = np.sum(image5) / 255
    sum6 = np.sum(image6) / 255
    
    # Memasukkan input ke dalam sistem kendali fuzzy
    robot_ctrl.input['velocity'] = sum5 * 255
    robot_ctrl.input['angle'] = (sum3 + sum6 - sum1 - sum4) * 90  # Perubahan pada rumus angle
    
    # Mengaktifkan sistem kendali fuzzy
    robot_ctrl.compute()
    
    # Mendapatkan output dari sistem kendali fuzzy
    left_output = robot_ctrl.output['left_speed']
    right_output = robot_ctrl.output['right_speed']
    
    # Menggerakkan robot
    robot.getDevice("left_wheel_hinge").setVelocity(left_output)
    robot.getDevice("right_wheel_hinge").setVelocity(right_output)
    
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.line(img_bgr, (0, boundary1), (w, boundary1), (0, 255, 255), thickness=1)
    cv2.line(img_bgr, (0, boundary2), (w, boundary2), (0, 255, 255), thickness=1)
    cv2.line(img_bgr, (int(w / 3), 0), (int(w / 3), h), (0, 255, 255), thickness=1)
    cv2.line(img_bgr, (int(2 * w / 3), 0), (int(2 * w / 3), h), (0, 255, 255), thickness=1)
    
    cv2.putText(img_bgr, '1', (int(w / 6), int(boundary1 / 2)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), thickness=1)
    cv2.putText(img_bgr, '2', (int(w / 2), int(boundary1 / 2)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), thickness=1)
    cv2.putText(img_bgr, '3', (int(5 * w / 6), int(boundary1 / 2)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), thickness=1)
    cv2.putText(img_bgr, '4', (int(w / 6), int((boundary1 + boundary2) / 2)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), thickness=1)
    cv2.putText(img_bgr, '5', (int(w / 2), int((boundary1 + boundary2) / 2)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), thickness=1)
    cv2.putText(img_bgr, '6', (int(5 * w / 6), int((boundary1 + boundary2) / 2)), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), thickness=1)
    
    if sum1 > 0:
        cv2.rectangle(img_bgr, (0, 0), (int(w / 3), boundary1), (0, 255, 0), thickness=1)
    if sum2 > 0:
        cv2.rectangle(img_bgr, (int(w / 3), 0), (int(2 * w / 3), boundary1), (0, 255, 0), thickness=1)
    if sum3 > 0:
        cv2.rectangle(img_bgr, (int(2 * w / 3), 0), (w, boundary1), (0, 255, 0), thickness=1)
    if sum4 > 0:
        cv2.rectangle(img_bgr, (0, boundary1), (int(w / 3), boundary2), (0, 255, 0), thickness=1)
    if sum5 > 0:
        cv2.rectangle(img_bgr, (int(w / 3), boundary1), (int(2 * w / 3), boundary2), (0, 255, 0), thickness=1)
    if sum6 > 0:
        cv2.rectangle(img_bgr, (int(2 * w / 3), boundary1), (w, boundary2), (0, 255, 0), thickness=1)
    
    cv2.imshow("Camera", img_bgr)
    cv2.waitKey(1)
    
cv2.destroyAllWindows()
