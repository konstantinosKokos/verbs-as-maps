from keras.layers import Input, Lambda, Dense, LSTM, Concatenate, Reshape, Dot
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam

def base_model(num_verbs, sentence_space=50, loss='binary_crossentropy',
               optimizer='rmsprop', verbose=False):
    """
    Returns the low layers of the core layer responsible for training the recurrent function
    """
    
    #### MAIN FUNCTIONAL LAYERS ####
    recurrent = LSTM(300, return_sequences=False, name='recurrent') # Accepts word sequences and returns sentence embeddings
    encoder = Dense(sentence_space, activation='tanh', kernel_regularizer='l2', name='encoder') # Compresses LSTM output
    cosine_function = Lambda(lambda x: 
                         ( 
                             (1 + K.sum( K.l2_normalize(x[:,:sentence_space], axis=-1) * K.l2_normalize(x[:,sentence_space:], axis=-1), axis=-1))/2
                         ),
                         (1,), name='cosine') # Accepts a horizontally stacked vector of two sentence embeddings and returns their sign-flipped cosine
    predictor = Dense(1, activation='sigmoid', name='predictor') # A trivial single-input single-output regression layer
    
    #### UTILITY LAYERS ####
    vconcat = Concatenate(axis=1, name='vconcat') # Stacks vector vertically
    hconcat = Concatenate(axis=-1, name='hconcat') # Stacks vectors horizontally
    to_scalar = Reshape((1,), name='to_scalar') # Converts a single-value tensor to a scalar
    add_dim = Reshape((1,300,), name='add_dim') # Increases the dimensionality of a vector
    
    #### INPUTS ####
    first_verb = Input(shape=(300,), name='first_verb')
    first_object = Input(shape=(300,), name='first_object')
    second_verb = Input(shape=(300,), name='second_verb')
    second_object = Input(shape=(300,), name='second_object')
    
    #### FLOW ####
    # ---- First, convert inputs to sequence ---- #
    first_vo = vconcat([add_dim(first_verb), add_dim(first_object)])
    second_vo = vconcat([add_dim(second_verb), add_dim(second_object)])
    # ---- Now pass them through the recurrent function ---- #
    first_vo = recurrent(first_vo)
    first_vo = encoder(first_vo)
    second_vo = recurrent(second_vo)
    second_vo = encoder(second_vo)
    # ---- Find their cosine distance ---- #
    cos = to_scalar(cosine_function(hconcat([first_vo, second_vo])))
    # ---- Predict whether they are paraphrases ---- #
    #prediction = predictor(cos)
    prediction = cos
    
    #### MODEL ####
    model = Model(inputs=[first_verb, first_object,
                          second_verb, second_object],
                  outputs=[prediction, first_vo, second_vo])
    

    if verbose:
        print('Training Network Summary:')
        model.summary()
    model.compile(loss = [loss, None, None],
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def evaluation_model(mapping_model, base_model, num_verbs, sentence_space=50, verbose=False):
    if mapping_model == None:
        verb_embedding = Input(shape=(300,))
        object_embedding = Input(shape=(300,))
        recurrent = base_model.get_layer('recurrent')
        encoder = base_model.get_layer('encoder')
        vconcat = base_model.get_layer('vconcat')
        add_dim = base_model.get_layer('add_dim')
        phrasal_representation = encoder(recurrent(vconcat([add_dim(verb_embedding), add_dim(object_embedding)])))
        model = Model(inputs=[verb_embedding, object_embedding], outputs=[phrasal_representation])
    else:
        verb_signature = Input(shape=(num_verbs,))
        object_embedding = Input(shape=(300,))
        to_matrix = mapping_model.get_layer('to_matrix')
        embedder = mapping_model.get_layer('embedder')
        apply = Dot(axes=-1)
        verb_map = to_matrix(embedder(verb_signature))
        phrasal_representation = apply([verb_map, object_embedding])
        model = Model(inputs=[verb_signature, object_embedding], outputs=[phrasal_representation])
    if verbose:
        print('Evaluation Network Summary:')
        model.summary()
    model.compile(loss = 'binary_crossentropy', optimizer='adam') # These compilation steps do not matter -  we won't train this model
    return model

def mapping_model(base_model, num_verbs, sentence_space=50, loss='binary_crossentropy', verbose=False, freeze=True, embedding_activation='relu',
                 lr=0.0005):
    """
    Returns the high level layers responsible for training the verb embeddings
    """
    
    #### METRIC WRAPPERS ####
    # ---- Returns percentage of positive predictions between a sentence pair ---- #
    def reconstruction_accuracy_wrapper(first_sentence, second_sentence):
        def reconstruction_accuracy(y_true, y_pred):
            merged_remaps = hconcat([first_sentence, second_sentence])
            cos_remap = cosine_function(merged_remaps)
            cos_remap = to_scalar(cos_remap)
            #return K.mean(K.equal(K.ones(cos_remap.shape[1]), K.round(predictor(cos_remap))), axis=-1)
            return K.mean(K.equal(K.ones(cos_remap.shape[1]), K.round(cos_remap)), axis=-1)
        return reconstruction_accuracy
    
    #### INIT ####
    # ---- Freeze the base model and recompile ---- #
    for layer in base_model.layers:
        layer.trainable = not freeze
    base_model.compile(loss = ['binary_crossentropy', None, None],
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    #### MAIN FUNCTIONAL LAYERS ####
    # ---- New layers ---- #
    embedder = Dense(units=(sentence_space * 300), activation=embedding_activation, name='embedder') # Produces a long vector out of verb signatures
    # ---- Maps to base layers ---- #
    recurrent = base_model.get_layer('recurrent')
    encoder = base_model.get_layer('encoder')
    cosine_function = base_model.get_layer('cosine')
    # predictor = base_model.get_layer('predictor') # Use if you want to perform regression on cosine similarity
    
    #### UTILITY LAYERS ####
    # ---- New layers ---- #
    to_matrix = Reshape((sentence_space, 300,), name='to_matrix') # Converts a long vector to a matrix
    apply = Dot(axes=-1)
    # ---- Maps to base layers ---- #
    to_scalar = base_model.get_layer('to_scalar')
    hconcat = base_model.get_layer('hconcat')
    vconcat = base_model.get_layer('vconcat')
    add_dim = base_model.get_layer('add_dim')
    
    #### INPUTS ####
    # ----  New inputs ---- #
    first_verb_signature = Input(shape=(num_verbs,), name='first_verb_signature')
    second_verb_signature = Input(shape=(num_verbs,), name='second_verb_signature')
    # ---- Maps to base inputs ---- #
    first_verb, first_object, second_verb, second_object = base_model.inputs[0:4]
    
    #### FLOW ####
    # ---- Construct tensor embeddings of verbs ---- #
    first_verb_map = to_matrix(embedder(first_verb_signature))
    second_verb_map = to_matrix(embedder(second_verb_signature))
    # ---- Reconstruct sentence embeddings ---- #
    first_remap = apply([first_verb_map, first_object])
    second_remap = apply([second_verb_map, second_object])
    # ---- Main output ---- #
    #prediction = predictor(to_scalar(cosine_function(hconcat([first_remap, second_remap]))))
    prediction = to_scalar(cosine_function(hconcat([first_remap, second_remap])))
    # ---- Auxilliary output ---- #
        # ---- Compute the recurrent embeddings ---- #
            # ---- First, convert inputs to sequence ---- #
    first_vo = vconcat([add_dim(first_verb), add_dim(first_object)])
    second_vo = vconcat([add_dim(second_verb), add_dim(second_object)])
            # ---- Now pass them through the recurrent function ---- #
    first_vo = recurrent(first_vo)
    first_vo = encoder(first_vo)
    second_vo = recurrent(second_vo)
    second_vo = encoder(second_vo)
        # ---- Additive losses ---- #
    #first_comparison = predictor(to_scalar(cosine_function(hconcat([first_remap, first_vo])))) 
    #second_comparison = predictor(to_scalar(cosine_function(hconcat([second_remap, second_vo]))))
    first_comparison = to_scalar(cosine_function(hconcat([first_remap, first_vo])))
    second_comparison = to_scalar(cosine_function(hconcat([second_remap, second_vo])))
    
    #### MODEL ####
    model = Model(inputs=[first_verb, first_object, first_verb_signature,
                          second_verb, second_object, second_verb_signature],
                  outputs=[prediction])
    
    if verbose:
        print('Embedding Network Summary:')
        model.summary()

    additive_loss = K.mean(K.square(first_comparison + second_comparison -2))
    model.add_loss(additive_loss)
    
    model.compile(loss = [loss],
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy', 
                           reconstruction_accuracy_wrapper(first_remap, first_vo),
                           reconstruction_accuracy_wrapper(second_remap, second_vo)])
    
    return model
