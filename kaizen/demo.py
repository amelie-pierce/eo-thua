import threading
import tkinter as tk
from tkinter import ttk, Entry, Button, Toplevel, messagebox
from functools import partial
import cv2
import dlib
import os
from PIL import Image, ImageTk
import face_recognition
from datetime import datetime, timedelta
import numpy as np
import json
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


employeeNameColumn = "Employee Name"
dateColumn = "Date"
checkinColumn = "Checkin"
checkoutColumn = "Checkout"
fontSize = 16

class EmployeeCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recongnition / Checkin / Checkout")
        root.geometry("1024x720")
        root.resizable(0, 0)

        #Config
        self.camera_checkin_running = True
        self.padding = 5

        self.config_time = 60        # seconds

        #Load Lib
        self.known_face_encodings = []
        self.known_face_names = []

        # Load know face
        self.load_known_faces()        
        self.cap = cv2.VideoCapture(0)
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.db = self.load_db()


        #Create Field

        self.tmp_employee_name_var = tk.StringVar()

        self.employee_name_var = tk.StringVar()
        self.employee_role_var = tk.StringVar()
        self.checkin_var = tk.StringVar()
        self.checkout_var = tk.StringVar()
        self.date_var = tk.StringVar()
        self.time_var = tk.StringVar()

        # Create UI

        # Create left frame for employee information
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, padx=(10, 10))
        
        title_label = tk.Label(left_frame, text="Infomation", font=('Helvetica', 24, 'bold'))
        title_label.grid(row=0, column=0, sticky="n", columnspan=2)

        # Employee Name Label and Entry
        self.employee_name_label = tk.Label(left_frame, text="Employee Name:", justify='left', font=('Helvetica', fontSize, 'bold'))
        self.employee_name_label.grid(row=1, column=0, padx=0, pady=5, sticky='w')
        self.employee_name_entry = tk.Label(left_frame, textvariable=self.employee_name_var, font=('Helvetica', fontSize, 'bold'))
        self.employee_name_entry.grid(row=1, column=1, padx=0, pady=5, sticky='w')

        # Role Label and Entry
        self.role_label = tk.Label(left_frame, text="Role:", justify='left', font=('Helvetica', fontSize, 'bold'))
        self.role_label.grid(row=2, column=0, padx=0, pady=5, sticky='w')
        self.role_entry = tk.Label(left_frame, textvariable=self.employee_role_var, font=('Helvetica', fontSize, 'bold'))
        self.role_entry.grid(row=2, column=1, padx=0, pady=5, sticky='w')

        # Create labels for time and date
        self.date_description_label = tk.Label(left_frame, text="Date:", justify='left', font=('Helvetica', fontSize, 'bold'))
        self.date_description_label.grid(row=3, column=0, padx=0, pady=5, sticky='w')
        self.date_label = tk.Label(left_frame, textvariable=self.date_var, font=('Helvetica', fontSize, 'bold'))
        self.date_label.grid(row=3, column=1, padx=0, pady=5, sticky='w')

        self.time_description_label = tk.Label(left_frame, text="Time:", justify='left', font=('Helvetica', fontSize, 'bold'))
        self.time_description_label.grid(row=4, column=0, padx=0, pady=5, sticky='w')
        self.time_label = tk.Label(left_frame, textvariable=self.time_var, font=('Helvetica', fontSize, 'bold'))
        self.time_label.grid(row=4, column=1, padx=0, pady=5, sticky='w')


        # # Checkin: 
        # self.checkin_label = tk.Label(left_frame, text="Checkin:", justify='left')
        # self.checkin_label.grid(row=5, column=0, padx=0, pady=5, sticky='w')
        # self.checkin_entry = tk.Label(left_frame, textvariable=self.checkin_var)
        # self.checkin_entry.grid(row=5, column=1, padx=0, pady=5, sticky='w')

        # self.checkout_label = tk.Label(left_frame, text="Checkout:", justify='left')
        # self.checkout_label.grid(row=6, column=0, padx=0, pady=5, sticky='w')
        # self.checkout_entry = tk.Label(left_frame, textvariable=self.checkout_var)
        # self.checkout_entry.grid(row=6, column=1, padx=0, pady=5, sticky='w')



        # Create a button to add a new user
        self.add_user_button = tk.Button(left_frame, text="Add New User", command=self.new_user_form, background='red', font=('Helvetica', fontSize))
        self.add_user_button.grid(row=7, column=0, pady=(20, 0), sticky='w', padx=(10, 0))
        self.add_user_button.grid()
        self.new_user_window = None


        self.display_attendance()
        # Create right frame for the camera feed
        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky='s')


        # Camera Feed Label
        self.camera_label = tk.Label(right_frame, text="Camera Feed")
        self.camera_label.grid(row=0, column=0)

        self.attendance_label = tk.Label(self.root, text="Camera Feessd")

        # Update Clock
        self.update_clock()

        self.reset_timer = None

        # Load Camera
        self.show_checkin_camera()

        # self.camera_thread = threading.Thread(target=camera_thread)
        # selfcamera_thread.start()


    # Attendance Log
    def display_attendance(self):
        self.filename = 'attendance.xlsx'
        self.df = pd.DataFrame(columns=[employeeNameColumn,dateColumn,checkinColumn,checkoutColumn])

        self.create_table()

        # Schedule the initial update and periodic updates
        self.update_table()

    def create_table(self):
        filename = 'attendance.xlsx'
        current_date = datetime.now().strftime("%d-%m-%Y")

        # Check if the Excel file exists
        if not os.path.exists(filename):
            # If the file doesn't exist, create an empty DataFrame and save it to create the Excel file
            df = pd.DataFrame(columns=[employeeNameColumn, dateColumn, checkinColumn, checkoutColumn])
            df.to_excel(filename, sheet_name=current_date, index=False)

        # Create a frame to contain the treeview
        table_frame = ttk.Frame(self.root)
        table_frame.grid(row=1, column=0, columnspan=2, pady=(20, 0), sticky='w', padx=(10, 0))

        # Create the ttk.Treeview inside the frame
        self.tree = ttk.Treeview(table_frame)
        self.tree.grid(row=0, column=0, sticky='w', padx=(0, 0), ipadx=200)  # Set ipadx to create the appearance of width

        self.tree["columns"] = [employeeNameColumn,dateColumn,checkinColumn,checkoutColumn]
        self.tree.heading("#0", text="", anchor="w")
        self.tree.column("#0", anchor="w", width=0)  # Placeholder column
        for column in  [employeeNameColumn,dateColumn,checkinColumn,checkoutColumn]:
            self.tree.heading(column, text=column, anchor="w")
            self.tree.column(column, anchor="w", width=150,)  # Adjust width as needed

        font_size = 16  # Change this to the desired font size
        style = ttk.Style()
        style.configure("Treeview", font=(None, font_size))
        style.configure("Treeview.Heading", font=(None, font_size))

        # Configure the column weight to make it expand to fill the available space
        table_frame.columnconfigure(0, weight=1)

    def update_table(self):
        try:
            current_date = datetime.now().strftime("%d-%m-%Y")
            # Read data from the Excel file
            self.df = pd.read_excel(self.filename,sheet_name=current_date)
            
            # Clear existing rows in the table
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Insert new rows into the table
            for index, row in self.df.iterrows():
                self.tree.insert("", "end", values=row.tolist())

        except pd.errors.EmptyDataError:
            # Handle the case when the Excel file is empty
            pass
        except Exception as e:
            print(f"Error updating table: {e}")

        # Schedule the next update after 1000 milliseconds (1 second)
        self.tree.after(1000, self.update_table)

    def save_attendance_data_to_excel(self):
        filename = 'attendance.xlsx'
        current_date = datetime.now().strftime("%d-%m-%Y")

        # Check if the Excel file exists
        if not os.path.exists(filename):
            # If the file doesn't exist, create an empty DataFrame and save it to create the Excel file
            df = pd.DataFrame(columns=[employeeNameColumn, dateColumn, checkinColumn, checkoutColumn])
            df.to_excel(filename, sheet_name=current_date, index=False)

        # Assuming self.checkin_employee_name_var.get() contains the current employee name
        employee_name = self.tmp_employee_name_var.get()

        if employee_name:
            current_time = datetime.now().strftime("%H:%M:%S")
            # Load existing data or create an empty DataFrame
            df = pd.read_excel(filename, sheet_name=current_date) if current_date in pd.ExcelFile(filename).sheet_names else pd.DataFrame(columns=[employeeNameColumn, dateColumn, checkinColumn, checkoutColumn])

            employee_data = df[df[employeeNameColumn] == employee_name]

            if not employee_data.empty:
                # Check if the date entry exists for the employee
                date_data = employee_data[employee_data[dateColumn] == current_date]
                if not date_data.empty:
                    # Update the existing date entry for the employee
                    df.loc[(df[employeeNameColumn] == employee_name) & (df[dateColumn] == current_date), checkoutColumn] = current_time
                else:
                    # Create a new entry for the employee with the current date
                    new_entry = {employeeNameColumn: employee_name, dateColumn: current_date, checkinColumn: current_time, checkoutColumn: None}
                    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            else:
                # Create a new entry for the employee with the current date
                new_entry = {employeeNameColumn: employee_name, dateColumn: current_date, checkinColumn: current_time, checkoutColumn: None}
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

            # Save the updated data to an Excel file
            with pd.ExcelWriter(filename, engine='openpyxl', if_sheet_exists='replace',mode='a') as writer:
                df.to_excel(writer, sheet_name=current_date, index=False,)

    # main func

    def stop_camera_checkin(self):
        self.camera_checkin_running = False

    def start_camera_checkin(self):
        self.camera_checkin_running = True
        self.show_checkin_camera()  # Start the camera feed

    def new_user_form(self):
        self.stop_camera_checkin()
        self.new_user_window = Toplevel(self.root)
        self.new_user_window.title("New User")

        left_frame = tk.Frame(self.new_user_window)
        left_frame.grid(row=0, column=0, padx=(10,10))

        title_label = tk.Label(left_frame, text="Add New User", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, sticky="n", pady=(10,10), columnspan=2)
        
        self.employee_name_label = tk.Label(left_frame, text="Employee Code:")
        self.employee_name_label.grid(row=1, column=0, padx=0, pady=5, sticky='w')
        self.employee_name_entry = Entry(left_frame)
        self.employee_name_entry.grid(row=1, column=1, padx=0, pady=5, sticky='w')

        self.employee_fullname_label = tk.Label(left_frame, text="Employee Name:")
        self.employee_fullname_label.grid(row=2, column=0, padx=0, pady=5, sticky='w')
        self.employee_fullname_entry = Entry(left_frame)
        self.employee_fullname_entry.grid(row=2, column=1, padx=0, pady=5, sticky='w')

        self.employee_role_label = tk.Label(left_frame, text="Employee Role:")
        self.employee_role_label.grid(row=3, column=0, padx=0, pady=5, sticky='w')
        self.employee_role_combobox = ttk.Combobox(left_frame, values=["Employee"])
        self.employee_role_combobox.grid(row=3, column=1, padx=0, pady=5, sticky='w')
        self.employee_role_combobox.set("Employee")

        self.capture_button = Button(left_frame, text="Add User", command=self.add_user)
        self.capture_button.grid(row=4, column=0, padx=0, pady=5, sticky='s', columns=2)

        self.capture_button = Button(left_frame, text="Close", command=self.on_new_user_window_close)
        self.capture_button.grid(row=5, column=0, padx=0, pady=5, columns=2, sticky='s')

        # Create right frame for the camera feed
        right_frame = tk.Frame(self.new_user_window)
        right_frame.grid(row=0, column=1)

        # Camera Feed Label
        self.camera_label_1 = tk.Label(right_frame, text="Camera Feed")
        self.camera_label_1.grid(row=0, column=0)

        self.show_camera_add_user()
        self.new_user_window.protocol("WM_DELETE_WINDOW", self.on_new_user_window_close)

    def add_user(self):
        filename = 'db.json'
        employee_name = self.employee_name_entry.get()

        count = 0
        total_images = 2
        

        # Delay time for take a photo
        last_second = -1
        delay_time_second = 1


        while count < total_images:
            ret, _frame = self.cap.read()

            if not ret:
                continue

            frame = cv2.flip(_frame, 1)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cur_second = datetime.now().second
        
            face_locations = face_recognition.face_locations(gray_frame, number_of_times_to_upsample = 1, model='hog')

            for (top, right, bottom, left) in face_locations:
                # Draw rectangles around detected faces
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 165, 0), 2)

                if(last_second < 0 or last_second >= 58):
                    last_second = cur_second
   
                if(cur_second - last_second >= delay_time_second):
                    # Save the detected face as an image with the given name
                    face_img = gray_frame[top:bottom, left:right]
                    count += 1
                    last_second = cur_second
                    image_path = self.generate_unique_filename(employee_name)
                    cv2.imwrite(image_path, face_img)
                    # Training one face in 1 frame
                break


        if employee_name in self.db:
                self.db[employee_name] = { 'role' :self.employee_role_combobox.get(), 'fullName': self.employee_fullname_entry.get() }
        else:
            self.db[employee_name] = { 'role': self.employee_role_combobox.get(), 'fullName': self.employee_fullname_entry.get() }
            # Save the updated data to the file
        with open(filename, 'w') as json_file:
            json.dump(self.db, json_file)  

        self.load_known_faces()
        self.db = self.load_db()
        messagebox.showinfo("Success", "Employee images captured and saved successfully!")
        
    def add_users_thread(self, employee_name, face, frame):
        if face: 
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            x -= self.padding
            y -= self.padding
            w += 2 * self.padding
            h += 2 * self.padding

            # Extract the face region from the frame
            face_region = frame[y:y+h, x:x+w]

            # Save the extracted face region as an image
            image_path = self.generate_unique_filename(employee_name)
            cv2.imwrite(image_path, face_region)

    def on_new_user_window_close(self):
        self.new_user_window.destroy()
        self.root.deiconify()
        self.root.focus_set()
        self.start_camera_checkin()

    def show_camera_add_user(self):
        ret, frame = self.cap.read()
        if ret:
            # Chuyển đổi khung hình sang dạng xám để nhận diện khuôn mặt
            frame = cv2.resize(frame, (550, 400))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(rgb_frame)
            if len(faces) > 0:
                # for (x, y, w, h) in faces:
                for face in faces:
                    self.reset_timer = None
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    x -= self.padding
                    y -= self.padding
                    w += 2 * self.padding
                    h += 2 * self.padding
                    # Vẽ hình chữ nhật xung quanh khuôn mặt
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Convert the frame to RGB format for display
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.camera_label_1.config(image=photo)
            self.camera_label_1.image = photo
            self.camera_label_1.after(30, self.show_camera_add_user)
        else:
            print("Error capturing frame from the camera")
    
    def show_checkin_camera(self):
        if self.camera_checkin_running:
            ret, frame = self.cap.read()
            if ret:
                # Chuyển đổi khung hình sang dạng xám để nhận diện khuôn mặt
                frame = cv2.resize(frame, (618, 365))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # faces = self.face_detector(rgb_frame)
                # faces = self.face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5)

                # Tìm các khuôn mặt trong frame
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                matches = []

                left = 0 
                bottom = 0

                face_distances = []
                process_this_frame = True
                threshold = 0.4

                if  process_this_frame:
                    if len(face_encodings) == 0:
                        self.reset_variables_after_delay()
                       

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                                                            
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                        
                        name = 'Unknown'
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)

                            if matches[best_match_index] and face_distances[best_match_index] < threshold:
                                name = self.known_face_names[best_match_index].split("_") 
                                if len(name) > 0:
                                    name = name[0]                    
                                    self.tmp_employee_name_var.set(name)
                                    self.get_user_info()
                                    self.save_attendance_data_to_excel()
                                    # process_this_frame = None

                                    
                            
                        if name == 'Unknown':
                            self.reset_variables_after_delay()
                        else:
                            self.reset_timer = None
                        
                        font = cv2.FONT_HERSHEY_DUPLEX
                                # Display the progress label on the live camera feed
                        # cv2.putText(frame, name, (10, 60), font, 1, (255, 165, 0), 2)
              

                # Convert the frame to RGB format for display
                photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.camera_label.config(image=photo)
                self.camera_label.image = photo
                self.camera_label.after(30, self.show_checkin_camera)
            else:
                print("Error capturing frame from the camera")

    # Load needed 
    def load_known_faces(self):
        images_folder = "data_set"
        self.known_face_encodings = []
        self.known_face_names = []

        def process_image(folder_name, filename):
            image_path = os.path.join(images_folder, folder_name, filename)
            image = face_recognition.load_image_file(image_path)
            small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)  # Resize to speed up processing
            face_encodings = face_recognition.face_encodings(small_image)

            if face_encodings:
                encoding = face_encodings[0]
                name = os.path.splitext(filename)[0]
                return encoding, name
            return None

        with ThreadPoolExecutor() as executor:
            results = []
            for folder_name in os.listdir(images_folder):
                folder_path = os.path.join(images_folder, folder_name)
                if os.path.isdir(folder_path):
                    folder_results = list(executor.map(partial(process_image, folder_name), [filename for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]))
                    results.extend(folder_results)

        for result in results:
            if result:
                encoding, name = result
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)
                
    def load_db(self):
        try:
            with open('./db.json') as json_file:
                return json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            # Create the file with an empty dictionary
            with open('./db.json', 'w') as json_file:
                json.dump({}, json_file)
            return {}
        

    # Helper func
    def get_user_info(self):
        self.db = self.load_db()
        employee_name = self.tmp_employee_name_var.get()
        if employee_name:
            if employee_name in self.db:
                employeeInDb = self.db[employee_name]
                if employeeInDb:
                    self.employee_role_var.set(employeeInDb['role'])
                    self.employee_name_var.set(employeeInDb['fullName'])

    def update_clock(self):
        current_datetime = datetime.now()

        current_time = current_datetime.strftime("%H:%M:%S")
        self.time_var.set(current_time)

        current_date = current_datetime.strftime("%d-%m-%Y")
        self.date_var.set(current_date)

        # Use after on the Tkinter widget (Label) to schedule updates
        self.time_label.after(1000, self.update_clock)

    def generate_unique_filename(self, base_filename):
        counter = 1
        folder_path = f"data_set/{base_filename}"
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

        image_path = f"{folder_path}/{base_filename}_{counter}.png"
        while os.path.exists(image_path):
            counter += 1
            image_path = f"{folder_path}/{base_filename}_{counter}.png"
        return image_path

    def reset_variables_after_delay(self):
        if self.reset_timer:
            self.reset_timer = True
        else:
            self.root.after(1000, self.reset_var)
            self.reset_timer = True

    def reset_var(self):
        self.checkout_var.set('')
        self.checkin_var.set('')
        self.employee_role_var.set('')
        self.employee_name_var.set('')
        self.tmp_employee_name_var.set('')

if __name__ == "__main__":
    root = tk.Tk()
    app = EmployeeCaptureApp(root)
    root.mainloop()
