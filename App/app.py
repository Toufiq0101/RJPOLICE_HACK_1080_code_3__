from flask import Flask, render_template
from flask_socketio import SocketIO
import os
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Replace 'your_csv_file.csv' with the actual name of your CSV file
csv_file_path = 'D:/ProjectX/App/templates/test_2.csv'

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    emit_csv_content()

def emit_csv_content():
    with open(csv_file_path, 'r') as file:
        csv_content = file.read()
        socketio.emit('csv_update', csv_content)

def watch_csv_file():
    last_modified_time = os.path.getmtime(csv_file_path)
    while True:
        time.sleep(1)
        current_modified_time = os.path.getmtime(csv_file_path)
        if current_modified_time > last_modified_time:
            last_modified_time = current_modified_time
            emit_csv_content()

if __name__ == '__main__':
    socketio.start_background_task(target=watch_csv_file)
    socketio.run(app, debug=True)

# import csv
# import ast

# from pyparsing import restOfLine

# def get_tracked_path(csv_file, target_number_plate):
#     with open(csv_file, 'r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             print(row)
#             number_plate = (row['number_plate'])
#             if target_number_plate in number_plate:
#                 tracked_path = (row['video_files'])
#                 return tracked_path
#     return None

# # Example usage
# csv_file_path = 'D:/ProjectX/App/introgations_results.csv'
# target_number_plate = 'JH10N9350'
# result = get_tracked_path(csv_file_path, target_number_plate)

# if result:
#     print(f"Tracked path for '{target_number_plate}': {result}")
# else:
#     print(f"No match found for '{target_number_plate}' in number plates.")
