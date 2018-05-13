# -*- coding: utf-8 -*-
"""
Algorithm stacking autoencoders layers ones after others.
"""

from keras.models import Model
from keras.layers import Dense, Input

class SuccessiveAutoencoder():
    """The algorithm stacks autoencoders one after others, with the idea to
    learn from the data higher level features.

    Attributes
    -----------------   
    - hidden_layers: number of hidden layers
    - units_nbr: number of neurons in each hidden layer
    - activation: type of activation to be used. Note that the activation of 
    the decoder is hardcoded within the class, and is a sigmoid
    - fine_tuning: if the algorithm train trains the full network once this one
    has been built. It can be 'y' or 'n'
    """
    
    def __init__(self, hidden_layers = 5, units_nbr = 5, activation = 'relu',
                 fine_tuning = 'y'):
        """Initialize the algorithm"""
        
        self.hidden_layers = hidden_layers
        self.units_nbr = units_nbr
        self.activation = activation
        self.fine_tuning = fine_tuning
        
    def create_model(self, X):
        """Create the inital autoencoder structure"""
        
        # Build simple autoencoder
        input_time_series = Input(shape = (X.shape[1], ), name = 'input')
        layer1 = Dense(units = self.units_nbr, activation = self.activation, 
                       name = 'hidden1')(input_time_series)
        decoded = Dense(units = X.shape[1], activation = 'sigmoid', name = 'output')(layer1)
        
        return Model(input_time_series, decoded)
    
    def add_layer(self, model, incr):
        """Add a new layer, and make previous layers not trainable"""
        
        # Set previous layers not trainable
        for layer in model.layers[:-1]:
            layer.trainable = False
           
        # Output of previous layer
        out = model.layers[-2].output
        
        # Add the new layer
        layer_new = Dense(units = self.units_nbr, activation = self.activation,
                          name = 'hidden' + str(incr))(out)
        
        decoded = model.layers[-1](layer_new)
        
        return Model(model.layers[0].input, decoded)

    def fit(self, X, epochs):
        """Run the algorithm, creating the autoencoder and calibrating it"""
        
        # Create the model and train it
        print('/ Training Hidden Layer 1')
        model = self.create_model(X)
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        
        h = model.fit(X, X, epochs = epochs, verbose = 0)
        print('Last loss: {}'.format(h.history['loss'][-1]))
        
        # Incrementally add layer, and train these new layers
        for incr in range(2, self.hidden_layers + 1):
            print('/ Training Hidden Layer {}'.format(str(incr)))
            model = self.add_layer(model, incr)
            model.compile(loss = 'mean_squared_error', optimizer = 'adam')
            
            h = model.fit(X, X, epochs = epochs, verbose = 0)
            print('Last loss: {}'.format(h.history['loss'][-1]))
          
        # If the user wants to run the calibration again over the complete model
        if self.fine_tuning == 'y':    
        
            # Final training
            print('/ Final Tuning')
            for layer in model.layers:
                layer.trainable = True
                
            model.compile(loss = 'mean_squared_error', optimizer = 'adam')
                
            h = model.fit(X, X, epochs = epochs, verbose = 0)
            print('Last loss: {}'.format(h.history['loss'][-1]))
        
        # Get rid of last layer, and stored the model
        model.layers.pop()
        
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        
        self.model = Model(model.layers[0].input, model.layers[-1].output)
        
        #self.model = model
            
    def predict(self, X):
        """Predict the compressed information from X"""
        
        return self.model.predict(X)
        
    
