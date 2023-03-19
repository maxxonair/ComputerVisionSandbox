// Basic demo for accelerometer readings from Adafruit MPU6050


/*
 * Project: StereoCam IMU 
 * 
 * Pin setup to connect MPU6050 to D1 mini
 * VLT -> 3.3V
 * GRD -> GRD
 * SDA -> Pin D2
 * SCL -> Pin D1 
 * 
 */
// -------------------------------------------------------------------------
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

/*
 * Unique device identifier
 * This identifier will be added to each serial message
 */
#define DEVICE_ID 605045772

/*
 * Maximum message identifier size 
 * max uint16
 */
#define MAX_MESSAGE_ID 65535
/*
 *            [Settings]
 */

/*
 * Initialize MPU6050 instance
 */
Adafruit_MPU6050 mpu;

/* 
 *  Create global variables for sensor events
 *  a    -> accelerometer measurements
 *  g    -> gyroscope measurements
 *  temp -> thermometer measurement
 */
sensors_event_t a, g, temp;

/*
 * Set meassurement and message frequency
 * 
 * unit: Hz 
 */
float printFrequency_Hz = 10;

/*
 * Flag enable print message that is sent via serial
 */
int enablePrintMessage = 1;


/* 
 *  Message ID
 */
uint16_t message_ID = 0;

/* 
 * ---------------------------------------------------------------------------------
 *                                  [SETUP]
 *          
 *          
 *          
 * ---------------------------------------------------------------------------------
 */
void setup(void) {
  initializeSerial();

  initializeImu();
  
  delay(100);
}

/* 
 * ---------------------------------------------------------------------------------
 *                                  [LOOP]
 *          
 *          
 *          
 * ---------------------------------------------------------------------------------
 */
void loop() {

  /* Get new sensor events with the readings */
  getImuSensorReadings();

  /* Send Serial message */
  serialAcc(enablePrintMessage);

  /* Delay to next measurement */
  delay(int(1/printFrequency_Hz * 1000));
}

/*
 * ---------------------------------------------------------------------------------
 *          [Service functions]
 * ---------------------------------------------------------------------------------
 */

void initializeSerial() {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens
}

void initializeImu() {
  Serial.println("Initialize Imu");

  // Try to initialize!
  // TODO: add error handling and send serial message if initialising fails
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    Serial.println("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    Serial.println("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    Serial.println("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    Serial.println("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    Serial.println("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    Serial.println("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    Serial.println("5 Hz");
    break;
  }

}

void incrementMessageId(){
  if ( message_ID >= MAX_MESSAGE_ID )
  {
    message_ID = 0;
  } else
  {
    message_ID = message_ID + 1;
  }
}

void getImuSensorReadings() {
  mpu.getEvent(&a, &g, &temp);
}

void serialAcc(int enablePrintMsg) {
  /* Compose message */
  String msg = "";
  incrementMessageId();
  /*
   * Sensor: N/A
   * Measured: N/A
   * Unit: N/A
   * Frame: N/A
   * Note: Unique device ID
   */
  msg.concat(String(DEVICE_ID));
  msg.concat(",");
  /*
   * Sensor: N/A
   * Measured: N/A
   * Unit: N/A
   * Frame: N/A
   * Note: Unique message ID
   */
  msg.concat(String(message_ID));
  msg.concat(",");
  /*
   * Sensor: Thermometer
   * Measured: temperature
   * Unit: deg celsius
   * Frame: N/A
   */
  msg.concat(String(temp.temperature));
  msg.concat(",");
  /*
   * Sensor: Accelerometers
   * Measured: Acceleration
   * Unit: m/s2
   * Frame: IMU6050 frame x/y/z
   */
  msg.concat(String(a.acceleration.x));
  msg.concat(",");
  msg.concat(String(a.acceleration.y));
  msg.concat(",");
  msg.concat(String(a.acceleration.z));
  msg.concat(",");
  /*
   * Sensor: Gyroscopes
   * Measured: angular velocity 
   * Unit: rad/s
   * Frame: IMU6050 frame x/y/z
   */
  msg.concat(String(g.gyro.x));
  msg.concat(",");
  msg.concat(String(g.gyro.y));
  msg.concat(",");
  msg.concat(String(g.gyro.z));
  if ( enablePrintMsg == 1 ){
    Serial.println(msg);
  }
  /* Send serial message */
  Serial.write(msg.c_str());
}