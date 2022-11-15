String command;


#define potentiometer_pin A0
#include <Servo.h>

Servo myservoP;
Servo myservoG;
Servo myservoO;
Servo myservoB;
Servo myservoY;
Servo myservoR;

void setup(){
  Serial.begin(9600);
  myservoP.attach(4);
  myservoG.attach(5);
  myservoO.attach(6); 
  myservoB.attach(7);
  myservoY.attach(8);
  myservoR.attach(9);
  
}

void loop(){
  
  myservoP.write(90);
  myservoG.write(90);
  myservoO.write(90);
  myservoB.write(90);
  myservoY.write(90);
  myservoR.write(90);
  
 // int reading = analogRead(potentiometer_pin);

 // Serial.println(reading);
  if (Serial.available()){
    command = Serial.readStringUntil('\n');
    command.trim();
   // Serial.println(command);
   // Serial.println(command);
    if (command.equals("P")){
    myservoP.write(110);
    delay(500);
    myservoP.write(80);
    delay(500);
   
  }
  else if (command.equals("G")){
    myservoG.write(55);
    delay(500);
    myservoG.write(75);
    delay(500);
  }
  else if( command.equals("O")){
    myservoO.write(110);
    delay(500);
    myservoO.write(90);
    delay(500);
  }
  else if( command.equals("B")){
    myservoB.write(115);
    delay(500);
    myservoB.write(90);
    delay(500);
  }
  else if( command.equals("Y")){
    myservoY.write(110);
    delay(500);
    myservoY.write(75);
    delay(500);
  }
  else if( command.equals("R")){
    myservoR.write(100);
    delay(500);
    myservoR.write(70);
    delay(500);
  }
  else if( command.equals("all")){
    myservoP.write(0);
    myservoG.write(0);
    myservoO.write(0);
    myservoB.write(0);
    myservoY.write(0);
    myservoR.write(0);
  }
  else if( command.equals("off")){
    myservoP.write(0);
    myservoG.write(0);
    myservoO.write(0);
    myservoB.write(0);
    myservoY.write(0);
    myservoR.write(0);
  }
  else{
    myservoP.write(0);
    myservoG.write(0);
    myservoO.write(0);
    myservoB.write(0);
    myservoY.write(0);
    myservoR.write(0);
  }
}

 delay(5);
}
