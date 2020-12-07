package com.example.elfandroid;

public class MyNdk {
    static {
        System.loadLibrary("MyLibrary");
    }

    public native String getString();
}
