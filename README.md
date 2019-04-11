# 3D Reconstruction of porous object using Machine learning

For my Research and Development unit (Bournemouth University Level6) I decided to 3D reconstruct a porous object such as a bone with the help of machine learning. The database that we used were mice tibia CT-scans.
I collaborated with a classmate, Lucien Hugueniot. 
We successfully made the machine learning algorithm predict the labels(porosity ranges and the health factor) that we created.
Our process was:
1. Finding the data set
2. See if the data set is large enough, if not we augment the data set by transforming the original images
3. Process the images to find the most accurate porosity factor of the bone, the health factor was already given in the data set. 
4. Label the data set based on the porosity factor and the health factor
5. Create the neural network 
6. Train the neural network and based on the results tweak the parameters or add layers to the the neural network
7. Depending on the accuracy of the neural network we 3D reconstruct the bone [This stage was not done due to time limits]

The last step wasn't done due to time limits, but further work would be 3D reconstruct the bone using openVDB and Houdini.
Further work would be to find a finer method to calculate the porosity factor so the labels would be more accurate and to get or create a bigger data set.

We used Keras, Tensorflow and Python to create the neural networks.

For further information there is a PDF report added to this project.
