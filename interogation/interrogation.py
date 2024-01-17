import csv
import datetime
from hmac import new
import os
from number_plate.main import process_video

def number_t(num, video):
    return True


def face_t(face, video):
    return True


def filter_by_middle_number(arr, lower_bound, upper_bound):
    return [
        element
        for element in arr
        if lower_bound <= get_middle_number(element) <= upper_bound
    ]


def get_middle_number(element):
    parts = element.split("_")
    return int(parts[1])


TR = []

def read_csv_file(file_path):

    with open(file_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader, None)
        if not any(row for row in csv_reader):
            print("The CSV file is empty.")
            return
        csvfile.seek(0)

        next(csv_reader, None)

        for row in csv_reader:
            footage_location = row[0]
            search_timeline = eval(row[1])

            # live searching
            if len(search_timeline) == 0:
                start_time_stamp = int(
                    footage_location.split("_")[1]
                )  # starting timestamp
                end_time = datetime.datetime.fromtimestamp(start_time_stamp)
                folder_name = end_time.strftime("%d-%m-%y")  # date
                end_time_stamp = int(
                    (end_time + datetime.timedelta(hours=1)).timestamp()
                )  # last timestamp
            face_search = eval(row[3])
            car_track = eval(row[4])
            found_in_current_video = False
            current_searching_file_time_stamp = start_time_stamp
            while current_searching_file_time_stamp <= end_time_stamp:
                data_storage_file_list = [
                    file
                    for file in os.listdir(f"D:/ProjectX/data_storage/{folder_name}")
                    if file.lower().endswith(".mp4")
                ]
                new_file_list = filter_by_middle_number(
                    data_storage_file_list, start_time_stamp, end_time_stamp
                )
                print(data_storage_file_list)
                if car_track and number_t(
                    car_track[0],
                    f"D:/ProjectX/data_storage/{folder_name}/{new_file_list[0]}",
                ):
                    TR.append(new_file_list[0])
                    found_in_current_video = True
                else:
                    if face_search and face_t(
                        face_search[0],
                        f"D:/ProjectX/data_storage/{folder_name}/{new_file_list[0]}",
                    ):
                        TR.append(new_file_list[0])
                        found_in_current_video = True
                if found_in_current_video == True:
                    current_searching_file_time_stamp = new_file_list[0].split("_")[1]
                    print(TR)
                    with open(
                        "D:/ProjectX/App/introgations_results.csv", mode="w", newline=""
                    ) as file:
                        print(f"--------------{TR}")
                        data = [
                            ["faces", "number_plate", "video_files"],
                            [f"{face_search}", f"{car_track}", f"{TR}"],
                        ]
                        writer = csv.writer(file)
                        writer.writerows(data)
                        found_in_current_video = False
                print(new_file_list)
                start_time_stamp = new_file_list[1]
                break
        print(TR)


csv_file_path = "D:/ProjectX/interogation/interrogation_data.csv"

read_csv_file(csv_file_path)

