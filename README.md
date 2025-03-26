## Music Generation with LSTM  

This project focuses on generating music using an LSTM (Long Short-Term Memory) neural network. The process involves preprocessing .krn music files, converting them into numerical data, and training an LSTM model to generate new music sequences.  

### *Key Features*  
- *Preprocessing:*  
   - Loads .krn music files and processes them into a uniform time-series representation.  
   - Transposes music to a common key (C major / A minor) for consistency.  
   - Handles polyphonic data and filters out notes with unacceptable durations.  

- *Data Preparation:*  
   - Converts the processed songs into integer mappings saved in mapping.json.  
   - Generates training sequences and one-hot encodes them for model training.  
   - Saves processed inputs and targets into separate .npy files for efficient loading.  

- *Model Training:*  
   - Utilizes an LSTM model to learn sequential patterns in the music data.  
   - Supports batch training to handle large datasets without exhausting memory.  
   - Continuously trains the model in batches to optimize memory usage.  

- *Model Saving:*  
   - The final trained model is saved as model.h5 for future inference.  

### *How to Run*  
1. *Preprocess the Dataset:*  
   Run the preprocessing script to convert .krn files into a unified format.  

2. *Train the Model:*  
   Use the training script to train the LSTM model in batches.  

3. *Generate Music:*  
   Use the trained model to generate new musical sequences.  

### *Dependencies*  
- Python 3.10  
- TensorFlow / Keras  
- Music21  
- NumPy
