#### Folder Structure

FI/FIDRT/TI are different intersections, but their folder structures and fields are basically the same.

##### video

Aerial video from the drone that has been stabilized, masked (except FIDRT), and down-sampled (to 10 FPS).



##### derived_data

These data are extracted and analyzed from the processed videos and are divided into 6 categories in total.



###### traj

The track point with turn annotation (after completing the lightweight framework).

| Field               | Category             | Description                                                  | Unit / Values                |
| :------------------ | :------------------- | :----------------------------------------------------------- | :--------------------------- |
| `frame`             | Tracking & Detection | Frame number corresponding to the observation (in the video of the same name). | -                            |
| `id`                | Tracking & Detection | Unique identifier for each tracked object, consistent across frames. | -                            |
| `type`              | Tracking & Detection | The classification of the object.                            | e.g., car, moped, pedestrain |
| `confidence`        | Tracking & Detection | The confidence score of the object detection, typically from 0 to 1. |                              |
| `isReal`            | Tracking & Detection | A flag indicating if the object is a real (1) detection or an interpolated one (0). | 0 or 1                       |
| `cx`                | Image Coordinates    | The x-coordinate of the bounding box center in the image plane. | pixels                       |
| `cy`                | Image Coordinates    | The y-coordinate of the bounding box center in the image plane. | pixels                       |
| `w`                 | Image Coordinates    | The width of the bounding box in the image plane.            | pixels                       |
| `h`                 | Image Coordinates    | The height of the bounding box in the image plane.           | pixels                       |
| `r`                 | Image Coordinates    | The rotation or orientation angle of the bounding box in the image plane. | radians                      |
| `cx_m`              | World Coordinates    | The x-coordinate of the object's center in a real-world coordinate system. | meters                       |
| `cy_m`              | World Coordinates    | The y-coordinate of the object's center in a real-world coordinate system. | meters                       |
| `w_m`               | World Coordinates    | The width of the object in real-world dimensions.            | meters                       |
| `h_m`               | World Coordinates    | The height of the object in real-world dimensions.           | meters                       |
| `length_med`        | World Coordinates    | The median length of the object detected in all frames in real-world dimensions. | meters                       |
| `width_med`         | World Coordinates    | The median width of the object detected in all frames in real-world dimensions. | meters                       |
| `w_med`             | Image Coordinates    | The median width of the object detected in all frames in pixels. | pixels                       |
| `h_med`             | Image Coordinates    | The median height of the object detected in all frames in pixels. | pixels                       |
| `time`              | Kinematics           | The timestamp of the observation.                            | seconds                      |
| `speed`             | Kinematics           | The instantaneous speed of the object.                       | m/s                          |
| `vx`                | Kinematics           | The velocity component of the object along the x-axis.       | m/s                          |
| `vy`                | Kinematics           | The velocity component of the object along the y-axis.       | m/s                          |
| `ax`                | Kinematics           | The acceleration component of the object along the x-axis.   | m/s²                         |
| `ay`                | Kinematics           | The acceleration component of the object along the y-axis.   | m/s²                         |
| `course`            | Kinematics           | The direction of travel angle in the Cartesian coordinate system calculated by kinematics. | radians                      |
| `yaw`               | Kinematics           | Target yaw angle after kinematic correction (along the long side). | radians                      |
| `smooth_cx`         | Smoothed Data        | The smoothed x-coordinate of the center point (mostly via a SG filter). | pixels                       |
| `smooth_cy`         | Smoothed Data        | The smoothed y-coordinate of the center point.               | pixels                       |
| `smooth_r`          | Smoothed Data        | The smoothed rotation angle.                                 | radians                      |
| `r_align`           | Smoothed Data        | The facing angle in the Cartesian coordinate system aligned to the world coordinate system. | radians                      |
| `speed_smooth`      | Smoothed Data        | The smoothed speed of the object.                            | m/s                          |
| `entry_direction`   | Turn Annotation      | The cardinal direction from which the object entered the scene. | N, S, E, W, Unknown          |
| `exit_direction`    | Turn Annotation      | The cardinal direction towards which the object exited the scene. | N, S, E, W, Unknown          |
| `overall_direction` | Turn Annotation      | Splicing of entry and exit directions.                       | e.g., N-E, N-N               |



###### map

Vector and raster maps with geographic coordinates, continuously updated.

* raster: TIFF format

  - only ending with ".tif", is the base map in the local projection coordinate system, and its Proj is:

    ```cmd
    +proj=tmerc +lon_0=0 +lat_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs
    ```

    

  - If it has the suffix wgs84, it is a raster map projected into the WGS84 coordinate system for vector map drawing.

* vector: OpenStreetMap vector map similar to the LaneLet2 standard (<mark>FIDRT is better, others are still under revision</mark>).



###### signal

Traffic light status data synchronized with each video, accurate to the second.

| Field        | Category          | Description                                                  | Unit / Values                                                |
| :----------- | :---------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `name`       | Scene             | Intersection scenes corresponding to traffic signals.        | FI, FIDRT, TI                                                |
| `direction`  | Signal Identifier | The cardinal direction of traffic flow controlled by this signal. | e.g., `'N'`, `'S'`, `'E'`, `'W'`                             |
| `turn`       | Signal Identifier | The turning movement controlled by the signal.               | s: Straight,<br />l: 'Left/U-turn',<br />r: 'Right',<br />p: 'Pedestrian (VRUs)' |
| `state`      | Signal State      | The active state or color of the traffic light for this signal group. | G, r, y                                                      |
| `begin_time` | Timing            | The timestamp when this signal state begins (in the video).  | seconds                                                      |
| `end_time`   | Timing            | The timestamp when this signal state ends.                   | seconds                                                      |
| `duration`   | Timing            | The total duration of this signal state (`end_time` - `begin_time`). | seconds                                                      |
| `cycle`      | Cycle Information | An identifier for the signal cycle to which this state belongs. | Counting starts from the time when the phase repetition occurs. |



###### flightLog

Drone flight logs with sensitive information such as plane coordinates removed.

| Field                           | Category            | Description                                                  | Unit / Values                               |
| :------------------------------ | :------------------ | :----------------------------------------------------------- | :------------------------------------------ |
| `time(millisecond)`             | Timestamp           | Elapsed time from the beginning of the flight log.           | ms                                          |
| `datetime`                      | Timestamp           | The local date and time of the record.                       | `YYYY-MM-DD HH:MM:SS`                       |
| `datetime(utc)`                 | Timestamp           | The UTC (Coordinated Universal Time) date and time.          | `YYYY-MM-DD HH:MM:SS`                       |
| `height_above_takeoff(feet)`    | Altitude            | The aircraft's relative height from the takeoff point.       | feet                                        |
| `altitude_above_seaLevel(feet)` | Altitude            | The aircraft's altitude relative to mean sea level (MSL).    | feet                                        |
| `height_sonar(feet)`            | Altitude            | Height measured by the downward-facing sonar/ultrasonic sensor. | feet                                        |
| `altitude(feet)`                | Altitude            | The primary altitude reading, typically the same as `height_above_takeoff`. | feet                                        |
| `ascent(feet)`                  | Altitude            | The total vertical distance climbed during the flight.       | feet                                        |
| `speed(mph)`                    | Speed & Distance    | The overall speed of the aircraft.                           | mph                                         |
| `xSpeed(mph)`                   | Speed & Distance    | The aircraft's velocity component along the East-West axis (positive is East). | mph                                         |
| `ySpeed(mph)`                   | Speed & Distance    | The aircraft's velocity component along the North-South axis (positive is North). | mph                                         |
| `zSpeed(mph)`                   | Speed & Distance    | The aircraft's vertical velocity (positive is up).           | mph                                         |
| `distance(feet)`                | Speed & Distance    | The horizontal distance from the home/takeoff point.         | feet                                        |
| `mileage(feet)`                 | Speed & Distance    | The total distance traveled by the aircraft during the flight. | feet                                        |
| `compass_heading(degrees)`      | Aircraft Attitude   | The direction the aircraft's nose is pointing, relative to magnetic North. | degrees (0-360)                             |
| `pitch(degrees)`                | Aircraft Attitude   | The aircraft's rotation around the side-to-side axis (nose up/down). | degrees                                     |
| `roll(degrees)`                 | Aircraft Attitude   | The aircraft's rotation around the front-to-back axis (tilting left/right). | degrees                                     |
| `gimbal_heading(degrees)`       | Gimbal Attitude     | The heading of the gimbal/camera relative to magnetic North. | degrees (0-360)                             |
| `gimbal_pitch(degrees)`         | Gimbal Attitude     | The pitch angle of the gimbal/camera (looking down/up).      | degrees                                     |
| `gimbal_roll(degrees)`          | Gimbal Attitude     | The roll angle of the gimbal/camera.                         | degrees                                     |
| `voltage(v)`                    | Power System        | The total voltage of the main flight battery.                | V                                           |
| `current(A)`                    | Power System        | The current being drawn from the battery.                    | A                                           |
| `battery_percent`               | Power System        | The remaining battery capacity as a percentage.              | %                                           |
| `battery_temperature(f)`        | Power System        | The temperature of the flight battery.                       | °F                                          |
| `voltageCell[1-6]`              | Power System        | The voltage of an individual battery cell (e.g., `voltageCell1`). | V                                           |
| `isPhoto`                       | System State & Logs | A flag indicating if a photo was taken at this timestamp.    | `true`, `false`                             |
| `isVideo`                       | System State & Logs | A flag indicating if video was being recorded at this timestamp. | `true`, `false`                             |
| `flycStateRaw`                  | System State & Logs | The raw numerical code for the flight controller's state.    | -                                           |
| `flycState`                     | System State & Logs | The decoded, human-readable state of the flight controller.  | e.g., `'GPS_Atti'`, `'P_Mode'`, `'Go_Home'` |
| `message`                       | System State & Logs | System-generated log messages or warnings.                   | -                                           |



###### route

Complete trajectory of motor vehicles (MVs) with information on time of entry and exit of intersections.

| Field               | Category                   | Description                                                  | Unit / Values                                       |
| :------------------ | :------------------------- | :----------------------------------------------------------- | :-------------------------------------------------- |
| `id`                | Identifier                 | Unique identifier for each individual MV trajectory.         | -                                                   |
| `in_time`           | Temporal                   | Timestamp when the MV enters the intersection.               | seconds                                             |
| `out_time`          | Temporal                   | Timestamp when the MV exits the intersection.                | seconds                                             |
| `type`              | Static Attribute           | The classification type of the MV.                           | e.g., car, bus, truck                               |
| `length_med`        | Physical Attribute         | The median length of the object detected in all frames in real-world dimensions. | meters                                              |
| `width_med`         | Physical Attribute         | The median width of the object detected in all frames in real-world dimensions. | meters                                              |
| `entry_direction`   | Trajectory Semantics       | The cardinal direction from which the MV entered the scene.  | e.g., `'N'`, `'S'`, `'E'`, `'W'`                    |
| `exit_direction`    | Trajectory Semantics       | The cardinal direction towards which the MV exited the scene. | e.g., `'N'`, `'S'`, `'E'`, `'W'`                    |
| `overall_direction` | Trajectory Semantics       | The dominant direction of the entire trajectory.             | e.g., `'N'`, `'S'`, `'E'`, `'W'`                    |
| `turn`              | Trajectory Semantics       | The turning movement performed by the MV at the intersection. | e.g., `'Straight'`, `'Left'`, `'Right'`, `'U-turn'` |
| `in_state`          | Traffic Signal Interaction | The traffic signal state when the MV crossed the entry line (e.g., stop line). | G, r, y                                             |
| `in_cycle`          | Traffic Signal Interaction | The signal cycle ID corresponding to the `in_state`.         | -                                                   |
| `out_state`         | Traffic Signal Interaction | The traffic signal state when the MV exit the intersection   | G, r, y                                             |
| `out_cycle`         | Traffic Signal Interaction | The signal cycle ID corresponding to the `out_state`.        | -                                                   |
| `in_delta`          | Traffic Signal Interaction | Time delta between the MV's arrival at the entry line and a key signal event (e.g., start of green). | seconds                                             |
| `out_delta`         | Traffic Signal Interaction | The time difference between the MV's exit from the intersection and the start of the corresponding signal light state. | seconds                                             |
| `geometry`          | Spatial Data               | The time difference between the MV's arrival at the entry line and the start of the corresponding signal light state. | e.g., `'LINESTRING (x1 y1, x2 y2, ...)'`            |



###### conflict

This is the traffic conflict situation after merging each video, and the time corresponding to minTTC is selected for recording.

| Field           | Category          | Description                                                  | Unit / Values                       |
| :-------------- | :---------------- | :----------------------------------------------------------- | :---------------------------------- |
| `Scene`         | Event Context     | The ID of the video where the conflict occurs.               | -                                   |
| `Frame`         | Event Context     | The specific frame number when the conflict reaches its minimum Time-to-Collision (MTTC). | -                                   |
| `Car 1 ID`      | Participants      | The unique identifier of the first vehicle involved in the conflict. | -                                   |
| `Car 2 ID`      | Participants      | The unique identifier of the second vehicle involved in the conflict. | -                                   |
| `Car 1 X`       | Participant State | The world x-coordinate of Car 1 at the moment of conflict.   | meters                              |
| `Car 1 Y`       | Participant State | The world y-coordinate of Car 1 at the moment of conflict.   | meters                              |
| `Car 2 X`       | Participant State | The world x-coordinate of Car 2 at the moment of conflict.   | meters                              |
| `Car 2 Y`       | Participant State | The world y-coordinate of Car 2 at the moment of conflict.   | meters                              |
| `diff`          | Conflict Metrics  | The minimum spatial distance between the two vehicles during the interaction. | meters                              |
| `TTC`           | Conflict Metrics  | Time-to-Collision: The time remaining until two vehicles would collide if they maintain their current speed and course. | seconds                             |
| `MTTC`          | Conflict Metrics  | Minimum Time-to-Collision: The minimum TTC value observed during the entire interaction, representing the point of highest risk. | seconds                             |
| `conflict_type` | Conflict Metrics  | The classification of the conflict based on vehicle trajectories. | rear end, sideswipe, angle, head on |

The calculation results of the conflict in the public dataset are being sorted out, and examples will be added later.

