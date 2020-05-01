package com.example.app_covid;
import android.os.Bundle;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.TextView;

//Importamos librería para agregar un delay.
import android.os.Handler;

import androidx.annotation.NonNull;

import org.tensorflow.lite.Interpreter;

import java.io.File;


public class MainActivity extends Activity {

    private BluetoothAdapter BTAdapter = BluetoothAdapter.getDefaultAdapter();
    /*
    //Interprete de TF Lite.
    Interpreter tflite;
    */

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        registerReceiver(receiver, new IntentFilter(BluetoothDevice.ACTION_FOUND));

        Button boton = (Button) findViewById(R.id.button1);
        boton.setOnClickListener(new OnClickListener(){
            public void onClick(View v) {
                BTAdapter.startDiscovery();
            }
        });

        Button boton_stop = (Button) findViewById(R.id.button2);
        boton_stop.setOnClickListener((new OnClickListener() {
            @Override
            public void onClick(View v) {
                BTAdapter.cancelDiscovery();
            }
        }));

        Button boton_reset = (Button) findViewById(R.id.button5);
        boton_reset.setOnClickListener((new OnClickListener() {
            @Override
            public void onClick(View v) {
                TextView reset = (TextView) findViewById(R.id.textView1);
                reset.setText("");
            }
        }));

       /*
        //First we must create the tflite object, loaded from the model file.
        try {
            tflite=new Interpreter(loadModelFile());
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        //When we click the infer button, we should do the inference.
        inferButon.setOnClickListener(new View.OnClickListener()){
            @Override
            public void onClick(View view) {
                float prediction = doInference(inputNumber.getText().toString());
                outputNumber.setText(Float.toString(prediction));
            }
        }
        */
    }


    private final BroadcastReceiver receiver = new BroadcastReceiver(){
        @Override
        public void onReceive(Context context, Intent intent) {

            String action = intent.getAction();
            if(BluetoothDevice.ACTION_FOUND.equals(action)) {
                int rssi = intent.getShortExtra(BluetoothDevice.EXTRA_RSSI,Short.MIN_VALUE);
                String name = intent.getStringExtra(BluetoothDevice.EXTRA_NAME);
                TextView rssi_msg = (TextView) findViewById(R.id.textView1);
                rssi_msg.setText(rssi_msg.getText() + name + " => " + rssi + "dBm\n");
            }
        }
    };
}
