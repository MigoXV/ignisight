def Correction(tempData):
    tempData = -7.8836 * 10 ** -6 * tempData ** 3 + 0.0176 * tempData ** 2 - 10.815 * \
                     tempData + 2725.9628
    return tempData