//int multiplexer_1_selectors_state[3] = {0, 0, 0}
int multiplexer_1_selectors_pins[3] = {2, 3, 4};

void setup() {  

  Serial.begin(9600); // Initialize the serial port
  // Set up the select pins as outputs:
  for (int i=0; i<3; i++)
  {
    pinMode(multiplexer_1_selectors_pins[i], OUTPUT);
    digitalWrite(multiplexer_1_selectors_pins[i], HIGH);
  }
  pinMode(A0, INPUT); // Set up Z as an input

}

void loop () {
 
  for (int count_1 = 0; count_1 < 5; count_1++) {

      digitalWrite(multiplexer_1_selectors_pins[0], bitRead(count_1,0));
      digitalWrite(multiplexer_1_selectors_pins[1], bitRead(count_1,1));
      digitalWrite(multiplexer_1_selectors_pins[2], bitRead(count_1,2));


  }
  Serial.println();
  delay(200);
}


void readAnalog(bool isLastPin) {
  int inputValue = analogRead(A0); // and read A0
  Serial.print(String(inputValue));
  if (!isLastPin) {
    Serial.print(",");
  }
}
