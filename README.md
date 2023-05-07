# CPSC_597
Final Project CPSC 597 Code 

Author: Madeline Smith 
Email: madeline.smith@csu.fullerton.edu 
CWID: 886664432

# Project Description

This project consists of 2 major parts, the first part of which is broken up into two sub-parts:

1. DNN Code, which is found in the dnn_code folder:
   
   1-A. Model construction using present evironmental data.

   1-B. Model construction using future environmental data. 

2. Code to build and run the web application, which can be found in the deploy_webapp folder. 

Model construction was based off of the following [paper](https://www.biorxiv.org/content/10.1101/744441v1) and [github repository](https://github.com/naturalis/trait-geo-diverse-dl).

In order to start the web application: please do the following 

1. Clone the gihub repository:
   
   ```gh repo clone maddie-aos/CPSC_597```

2. From the terminal switch to the deploy_webapp directory:
   
   ```cd deploy_webapp```

3. Activate the virtual environment:
   
   ```source env/bin/activate ```

4. Install flask if you do not have it:
   
   ```pip install flask```

5. Run the flask application:
   
   ```flask run``` 

If the application is running properly the following message should be displayed on the terminal: 
```bash
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 2023-05-07 12:17:24.572421: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
 To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
``` 
6. To stop the application, run the following:
7. 
   ```Ctrl + C``` : this stops the server.

   ```deactivate```: this deactivates the virtual environment.




