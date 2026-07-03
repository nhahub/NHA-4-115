#include "GPIO_driver.h"
#include "Motor_driver.h"
#include "ADC_driver.h"
#include "Battery_indicator_driver.h"
#include "Seven_segment_driver.h"

typedef enum { STOPPED=0, FORWARD=1, BACKWARD=2 } State;

/* globals */
Motor_Config      motor;
Battery_Indicator battery;
SevenSeg_Single   seg1;
SevenSeg_Double   seg2;
GPIO_PinConfig    sw1, sw2, sw3;

void init_all(void) {
    uint8_t i;

    /* switches RA1, RA2, RA3 */
    sw1.port=PORT_A; sw1.pin=1; sw1.direction=GPIO_INPUT; GPIO_PinInit(&sw1);
    sw2.port=PORT_A; sw2.pin=2; sw2.direction=GPIO_INPUT; GPIO_PinInit(&sw2);
    sw3.port=PORT_A; sw3.pin=3; sw3.direction=GPIO_INPUT; GPIO_PinInit(&sw3);

    /* motor RD0, RD1 */
    motor.in1.port=PORT_D; motor.in1.pin=0;
    motor.in2.port=PORT_D; motor.in2.pin=1;
    Motor_Init(&motor);

    /* single 7-seg RC0-RC6 */
    for(i=0;i<7;i++){seg1.seg[i].port=PORT_C; seg1.seg[i].pin=i;}
    SevenSeg_Seg_Init(&seg1);

    /* double 7-seg RB0-RB6, sel RD2/RB7 */
    for(i=0;i<7;i++){seg2.seg[i].port=PORT_B; seg2.seg[i].pin=i;}
    seg2.sel[0].port=PORT_D; seg2.sel[0].pin=2;
    seg2.sel[1].port=PORT_B; seg2.sel[1].pin=7;
    SevenSeg_Double_Init(&seg2);

    /* battery LEDs RC5, RD3, RD4, RD5 */
    battery.led[0].port=PORT_C; battery.led[0].pin=5;
    battery.led[1].port=PORT_D; battery.led[1].pin=3;
    battery.led[2].port=PORT_D; battery.led[2].pin=4;
    battery.led[3].port=PORT_D; battery.led[3].pin=5;
    Battery_Init(&battery);
}

void main(void) {
    uint8_t  s1p=0,s2p=0,s3p=0, s1c,s2c,s3c;
    uint16_t btimer=0, raw;
    State    state=STOPPED;

    ADC_Init();
    init_all();
    Motor_Stop(&motor);
    SevenSeg_Single_Show(&seg1,'S');
    raw=ADC_Read();
    SevenSeg_Double_Show(&seg2, ADC_x10(raw));
    Battery_SetLevel(&battery, ADC_level(raw));

    while(1){
        s1c=GPIO_ReadPin(&sw1);
        s2c=GPIO_ReadPin(&sw2);
        s3c=GPIO_ReadPin(&sw3);

        if(s1c&&!s1p){state=FORWARD;  Motor_RunForward(&motor);  SevenSeg_Single_Show(&seg1,'F');}
        if(s2c&&!s2p){state=BACKWARD; Motor_RunBackward(&motor); SevenSeg_Single_Show(&seg1,'b');}
        if(s3c&&!s3p){state=STOPPED;  Motor_Stop(&motor);        SevenSeg_Single_Show(&seg1,'S');}

        s1p=s1c; s2p=s2c; s3p=s3c;

        raw=ADC_Read();
        SevenSeg_Double_Show(&seg2, ADC_x10(raw));

        btimer++;
        if(btimer>=3000){ btimer=0; Battery_SetLevel(&battery,ADC_level(raw)); }

        Delay_ms(1);
    }
}
