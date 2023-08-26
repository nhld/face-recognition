from deepface import DeepFace
import cv2
import os
import time

def get_folder_name(file_path):
    parent_dir = os.path.dirname(file_path)
    folder_name = os.path.basename(parent_dir)
    return folder_name

face_data_folder_path = "face_data/"

if os.path.exists("new_face.jpg"):
    os.remove("new_face.jpg")

if os.path.exists("face_data/representations_facenet512.pkl"):
    os.remove("face_data/representations_facenet512.pkl")

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 5)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video_capture.isOpened():
    print("Error: cant open camera.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: cant not capture frame from camera.")
        video_capture.release()
        break

    faces = DeepFace.extract_faces(img_path = frame, target_size = (224, 224), detector_backend = 'mtcnn', enforce_detection=False)
    
    for face in faces:
        bounding_box = face["facial_area"]
        x = bounding_box["x"]
        y = bounding_box["y"]
        w = bounding_box["w"]
        h = bounding_box["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        detected_face = frame[y:y+h, x:x+w]

        if detected_face.size > 0 and detected_face.shape != frame.shape:
            cv2.imwrite("new_face.jpg", detected_face)

            result = DeepFace.find(img_path="new_face.jpg", db_path=face_data_folder_path, enforce_detection=False, model_name='Facenet512')
            if len(result[0]["identity"]) != 0:
                filepath = str(result[0]["identity"])
                split_filepath = filepath.split()
                face_folder = str(get_folder_name(split_filepath[1]))
                
                if face_folder != '':
                    cv2.putText(frame, face_folder, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f'{face_folder} checked in.', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
            else:
                if detected_face.any():
                
                    cv2.putText(frame, "Person not recognized.", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    current_time = time.localtime()
                    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)

                    initial_folders_starting_with_p = [folder for folder in os.listdir(face_data_folder_path) if os.path.isdir(os.path.join(face_data_folder_path, folder)) and folder.startswith("p")]
                    num_folders_starting_with_p = len(initial_folders_starting_with_p)

                    new_face_path = os.path.join(face_data_folder_path, f'p_{num_folders_starting_with_p + 1}')
                    print(num_folders_starting_with_p)
                    print(new_face_path)
                    
                    if not os.path.exists(os.path.join(face_data_folder_path, new_face_path)):
                        os.makedirs(new_face_path)
                    else:
                        pass
                    new_filename = os.path.join(new_face_path, f"image_{formatted_time}.jpg")
                    cv2.imwrite(new_filename, detected_face)

                    if os.path.exists("face_data/representations_facenet512.pkl"):
                        os.remove("face_data/representations_facenet512.pkl")
                            
                else:
                    cv2.putText(frame, " ", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()