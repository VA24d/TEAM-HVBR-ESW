// Ultrasonic code 
// File: esp32_ultrasonic.ino

const int trigPin = 32;  // Trigger Pin
const int echoPin = 33; // Echo Pin

void setup() {
  Serial.begin(115200);  // Start Serial communication at 9600 bps
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  long duration;
  float distance;

  // Clear the trigger pin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  // Set the trigger pin HIGH for 10 microseconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Read the echo pin duration in microseconds
  duration = pulseIn(echoPin, HIGH);

  // Calculate the distance in cm (assuming speed of sound is 343 m/s)
  distance = duration * 0.0343 / 2;

  // Send distance to Serial
  Serial.println(distance);

  delay(500);  // Adjust as needed
}