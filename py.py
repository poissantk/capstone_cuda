import serial

if __name__== '__main__':
    ser = serial.Serial('/dev/ttyACM0',9600,timeout=1)
    ser.flush()
    
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(line)
            if float(line)<=200:
                ser.write(b"E\n")
            elif float(line)>200 and float(line)<=400:
                ser.write(b"A\n")
            elif float(line)>400 and float(line)<=600:
                ser.write(b"D\n")
            elif float(line)>600:
                ser.write(b"G\n")
            else:
                ser.write(b"all\n")


