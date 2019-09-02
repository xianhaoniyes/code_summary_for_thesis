import numpy as np
from keras.models import load_model
from keras import backend as K


vectors = np.load('new_vectors_matrix.npy')
person_to_remain = np.load('new_person_to_remain.npy')
vectors = vectors[person_to_remain]


neural_vectors = []
for f in range(0, 5):
    model = load_model('model_'+str(f)+'.h5')
    get_dense_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])
    fold_neural_vectors = []
    for i in range(0, len(vectors)):
        layer_output = get_dense_layer_output([np.expand_dims(vectors[i], axis=3), 0])[0]

        fold_neural_vectors.append(layer_output)

    np.save('fold_vectors_'+str(f)+'.npy', fold_neural_vectors)





