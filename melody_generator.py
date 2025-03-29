import tensorflow.keras as keras # type:ignore
from preprocessing import SEQUENCE_LENGTH, MAPPING_PATH
import json
import music21 as m21
import numpy as np

class MelodyGenerator:

    def __init__(self, model_path = 'model1.h5'):
        
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, 'rb') as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    
    def generate_melody(self, seed, num_steps, max_seq_len, temprature):
        
        ## create seed with start symbol 

        seed = seed.split()
        melody = seed 
        seed = self._start_symbols + seed 

        ## map seed to integers

        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            ## limit the seed to max_sequence length 
            seed = seed[-max_seq_len:]

            ## one-hot encode the seed 
            seed_onehot = keras.utils.to_categorical(seed, num_classes = len(self._mappings))
            # (1, max_seq_len, num of symbols in vocabulary )
            seed_onehot = seed_onehot[np.newaxis, ...]

            ## make a prediction 
            probabilites = self.model.predict(seed_onehot)[0]

            output_int = self._sample_with_temprature(probabilites, temprature)

            ## update seed
            seed.append(output_int)

            ## map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            ## check whenever we're at the end of a melody

            if output_symbol == "/":
                break

            ## update melody 
            melody.append(output_symbol)
    
        return melody
    


    
    def _sample_with_temprature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature

        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index
    
    def save_melody(self, melody, step_duration = 1, format = 'midi', filename = 'melody.midi'): 
        
        ## create music21 stream
        stream = m21.stream.Stream()  


        ## parse all the symbols in melody and create notes/rests objects 
        start_symbol = None
        step_counter = 1
        
        for i, symbol in enumerate(melody): 
            ## handle case in which we have notes/rests
            if symbol != '_' or i+1 == len(melody):
                
                ## ensure that we're dealing with note/rests beyond first one
                if start_symbol is not None: 
                    
                    quarter_length_duration = step_duration * step_counter
                    # handle note
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength = quarter_length_duration)

                    # handle rest
                    else: 
                        m21_event = m21.note.Note(int(start_symbol), quarterLength = quarter_length_duration)

                    stream.append(m21_event)

                    # reset step counter

                    step_counter = 1

                start_symbol = symbol 

                
            ## handle case in which we have prolongation sign '_'
            else: 
                step_counter += 1

        ## write the m21 stream to a midi file 
        stream.write(format, filename)

if __name__ == '__main__':
    
    mg = MelodyGenerator()
    seed = '67 _ 67 _ 67 _ _ 65 64 _ 64 _ 60 '

    melody = mg.generate_melody(seed, 250, SEQUENCE_LENGTH, 1.7)
    print(melody)

    mg.save_melody(melody)



