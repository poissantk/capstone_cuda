String command;


#define potentiometer_pin A0
#include <Servo.h>

Servo myservoE;
Servo myservoA;
Servo myservoD;
Servo myservoG;
Servo myservoB;
Servo myservoE1;

void setup(){
  Serial.begin(9600);
  myservoE.attach(7);
  myservoA.attach(8);
  myservoD.attach(9); 
  myservoG.attach(10);
  myservoB.attach(11);
  myservoE1.attach(12);
  
}

void loop(){
  
  myservoE.write(90);
  myservoA.write(90);
  myservoD.write(90);
  myservoG.write(90);
  myservoB.attach(90);
  myservoE1.attach(90);
  
 // int reading = analogRead(potentiometer_pin);

 // Serial.println(reading);
  if (Serial.available()){
    command = Serial.readStringUntil('\n');
    command.trim();
   // Serial.println(command);
    if (command.equals("E")){
    myservoE.write(110);
    delay(500);
    myservoE.write(80);
    delay(500);
   
  }
  else if (command.equals("A")){
    myservoA.write(110);
    delay(500);
    myservoA.write(80);
    delay(500);
  }
  else if( command.equals("D")){
    myservoD.write(110);
    delay(500);
    myservoD.write(90);
    delay(500);
  }
  else if( command.equals("G")){
    myservoG.write(115);
    delay(1000);
    myservoG.write(90);
  }
  else if( command.equals("B")){
    myservoB.write(110);
    delay(500);
    myservoB.write(75);
    delay(500);
  }
  else if( command.equals("E1")){
    myservoE1.write(100);
    delay(500);
    myservoE1.write(70);
    delay(500);
  }
  else if( command.equals("all")){
    myservoE.write(0);
    myservoA.write(0);
    myservoD.write(0);
    myservoG.write(0);
    myservoB.write(0);
    myservoE1.write(0);
  }
  else if( command.equals("off")){
    myservoE.write(0);
    myservoA.write(0);
    myservoD.write(0);
    myservoG.write(0);
    myservoB.write(0);
    myservoE1.write(0);
  }
  else{
    myservoE.write(0);
    myservoA.write(0);
    myservoD.write(0);
    myservoG.write(0);
    myservoB.write(0);
    myservoE1.write(0);
  }
}

delay(1000);
}
