Official ROS Documentation
--------------------------

🚧 **WIP ([@pokusew](https://github.com/pokusew)): Improving code, compatibility with the latest ROS 2 versions,
implementing missing functionalities, rethinking internal workings.**

Converted to ROS2 from the package `xycar_imu`.

A much more extensive and standard ROS-style version of this documentation can be found on the ROS wiki at:

http://wiki.ros.org/xycar_imu


Install and Configure ROS Package
---------------------------------
1) Download code and install: 

    ```
    $ git clone https://github.com/klintan/xycar_imu.git
    $ cd ..
    $ colcon build --symlink-install
    ```


Install Arduino firmware
-------------------------
1) For SEN-14001 (9DoF Razor IMU M0), you will need to follow the same instructions as for the default firmware on https://learn.sparkfun.com/tutorials/9dof-razor-imu-m0-hookup-guide and use an updated version of SparkFun_MPU-9250-DMP_Arduino_Library from https://github.com/lebarsfa/SparkFun_MPU-9250-DMP_Arduino_Library (an updated version of the default firmware is also available on https://github.com/lebarsfa/9DOF_Razor_IMU).

2) Open ``src/Razor_AHRS/Razor_AHRS.ino`` in Arduino IDE. Note: this is a modified version
of Peter Bartz' original Arduino code (see https://github.com/ptrbrtz/razor-9dof-ahrs). 
Use this version - it emits linear acceleration and angular velocity data required by the ROS Imu message

3) Select your hardware here by uncommenting the right line in ``src/Razor_AHRS/Razor_AHRS.ino``, e.g.

<pre>
// HARDWARE OPTIONS
/*****************************************************************/
// Select your hardware here by uncommenting one line!
//#define HW__VERSION_CODE 10125 // SparkFun "9DOF Razor IMU" version "SEN-10125" (HMC5843 magnetometer)
//#define HW__VERSION_CODE 10736 // SparkFun "9DOF Razor IMU" version "SEN-10736" (HMC5883L magnetometer)
#define HW__VERSION_CODE 14001 // SparkFun "9DoF Razor IMU M0" version "SEN-14001"
//#define HW__VERSION_CODE 10183 // SparkFun "9DOF Sensor Stick" version "SEN-10183" (HMC5843 magnetometer)
//#define HW__VERSION_CODE 10321 // SparkFun "9DOF Sensor Stick" version "SEN-10321" (HMC5843 magnetometer)
//#define HW__VERSION_CODE 10724 // SparkFun "9DOF Sensor Stick" version "SEN-10724" (HMC5883L magnetometer)
</pre>

4) Upload Arduino sketch to the Sparkfun 9DOF Razor IMU board


Configure
---------
In its default configuration, ``xycar_imu`` expects a yaml config file ``xycar_imu.yaml`` with:
* USB port to use
* Calibration parameters

An example``razor.yaml`` file is provided.
Copy that file to ``xycar_imu.yaml`` as follows:

    $ roscd xycar_imu/config
    $ cp razor.yaml xycar_imu.yaml

Then, edit ``xycar_imu.yaml`` as needed

The provided configuration publishes all data exposed by the bundled firmware:

* ``imu`` (``sensor_msgs/msg/Imu``): orientation, angular velocity, and linear acceleration
* ``mag`` (``sensor_msgs/msg/MagneticField``): firmware-reported magnetic field,
  converted from milligauss to tesla

Set ``publish_magnetometer`` to ``false`` if the separate magnetic-field topic is not needed.
The provided magnetometer limits are generic defaults, not calibration values for a specific
board. Treat the default ``mag`` values as relative measurements and calibrate the installed IMU
before using their absolute magnitude or heading for navigation.

Launch
------
Publisher:

	$ ros2 launch xycar_imu xycar_imu.launch.py

Publisher and 3D visualization:

	$ ros2 launch xycar_imu xycar_imu_and_display.launch.py

3D visualization with diagnostics tools (publisher must already be running):

	$ ros2 launch xycar_imu razor-pub-diags.launch.py

3D visualization only:

	$ ros2 launch xycar_imu xycar_imu_display.launch.py

Publisher with RViz:

	$ ros2 launch xycar_imu xycar_imu_viewer.launch.py

The publisher and visualization executables can also be run directly:

	$ ros2 run xycar_imu imu_node

	$ ros2 run xycar_imu display_3D_visualization_node


Calibrate
---------
For best accuracy, follow the tutorial to calibrate the sensors:

http://wiki.ros.org/xycar_imu

An updated version of Peter Bartz's magnetometer calibration scripts from https://github.com/ptrbrtz/razor-9dof-ahrs is provided in the ``magnetometer_calibration`` directory.

Update ``my_razor.yaml`` with the new calibration parameters.

Dynamic Reconfigure
-------------------
Not yet supported in the ROS2 version

After having launched the publisher with one of the launch commands listed above, 
it is possible to dynamically reconfigure the yaw calibration.

1) Run:

    $ rosrun rqt_reconfigure rqt_reconfigure 
    
2) Select ``imu_node``. 

3) Change the slider to move the calibration +/- 10 degrees. 
If you are running the 3D visualization you'll see the display jump when the new calibration takes effect.

The intent of this feature is to let you tune the alignment of the AHRS to the direction of the robot driving direction, so that if you can determine that, for example, the AHRS reads 30 degrees when the robot is actually going at 35 degrees as shown by e.g. GPS, you can tune the calibration to make it read 35. It's the compass-equivalent of bore-sighting a camera.
