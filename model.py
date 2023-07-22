'''
GNet
GNet_Fusion
'''

'''
Base
'''
def SpeE(X):
    L = CNA3D(16, (1, 1, 3), 1, 'same', X, 'gelu')
    return L

def SpaE(X):
    L = CNA3D(16, (3, 3, 1), 1, 'same', X, 'gelu')
    return L

'''
S2CLF2
'''
def Intra_Level_Fusion(X1, X2):
    S1 = tf.split(X1, num_or_size_splits=2, axis=4)
    S2 = tf.split(X2, num_or_size_splits=2, axis=4)
    L1 = layers.add([S1[0], S2[0]])
    L2 = layers.concatenate([S1[1], L1])
    L3 = layers.concatenate([L2, S2[1]])

    return L3

def Inter_Level_Fusion(X1, X2):
    L1 = layers.multiply([X1, X2])
    A = layers.Activation('sigmoid')(L1)
    Add1 = layers.multiply([A, X1])
    Add2 = layers.multiply([A, X2])
    L2 = layers.concatenate([Add1, Add2])

    return L2


def S2CLF2(Spe_L, Spa_L, Spe_H, Spa_H):
    L1 = Intra_Level_Fusion(Spe_L, Spa_L)
    L2 = Intra_Level_Fusion(Spe_H, Spa_H)
    L3 = Inter_Level_Fusion(L1, L2)

    return L3

'''
Model Definition
'''
def GNet_Fusion(X):
    L11_2 = SpaE(X)
    L11_3 = SpeE(X)

    L12_2 = SpaE(L11_2)
    L12_3 = SpeE(L12_2)

    L13_2 = SpaE(L12_2)
    L14_2 = SpaE(L13_2)
    L13_3 = SpeE(L12_2)

    L21_2 = SpaE(L11_3)
    L21_3 = SpeE(L11_3)
    Add22 = layers.add([L12_3, L21_2])
    L22_2 = SpaE(Add22)
    L22_3 = SpeE(Add22)
    Add23 = layers.add([L13_3, L22_2])
    L23_2 = SpaE(Add23)
    L23_3 = SpeE(Add23)
    Add24 = layers.add([L23_3, L14_2])
    L24_2 = SpaE(Add24)

    L31_2 = SpaE(L21_3)
    L31_3 = SpeE(L21_3)
    Add32 = layers.add([L31_2, L22_3])
    L32_2 = SpaE(Add32)
    L32_3 = SpeE(Add32)
    Add33 = layers.add([L32_3, L23_2])
    L33_2 = SpaE(Add33)
    L33_3 = SpeE(Add33)
    # Add34 = layers.add([L33_3, L24_2])
    F1 = S2CLF2(L12_2, L12_3, L24_2, L33_3)
    L34_2 = SpaE(F1)

    L41_3 = SpeE(L31_3)
    Add42 = layers.add([L41_3, L32_2])
    L42_3 = SpeE(Add42)
    # Add43 = layers.add([L42_3, L33_2])
    F2 = S2CLF2(L21_2, L21_3, L33_2, L42_3)
    L43_3 = SpeE(F2)
    # Add44 = layers.add([L34_2, L43_3])
    F3 = S2CLF2(L11_2, L11_3, L34_2, L43_3)

    L4 = CNA3D(32, (3, 3, 3), 1, 'same', F3, 'gelu')

    return L4

def Model(input_shape=(21, 21, 15, 1), classes=16):
    inputs = layers.Input(shape=input_shape)
    L = GNet_Fusion(inputs)
    GAP = layers.GlobalAveragePooling3D()(L)
    outputs = layers.Dense(units=classes, activation='softmax')(GAP)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='GNet')
    return model
