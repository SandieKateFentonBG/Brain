STR_FEATURES = ['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding', 'BREEAM Rating']
INT_FEATURES = ['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN/m2)']
FEATURES_NAMES = STR_FEATURES + INT_FEATURES
INT_OUTPUTS = ['Calculated Total tCO2e', 'Calculated tCO2e/m2']
INT_PARAMETERS = {'GIFA (m2)': (16, 8, 1), 'Storeys': (5, 1, 1), 'Typical Span (m)': (7, 0, 5), 'Typ Qk (kN/m2)': (5, 0, 2),
                  'Calculated Total tCO2e': (15, 7, 1), 'Calculated tCO2e/m2': (11, 4, 1000)}#(siz,cut, precision)
"""
STR_PARAMETERS = {'Sector': (7), 'Type': (5), 'Basement': (4), 'Foundations': (7),
                  'Ground Floor': (6), 'Superstructure': (12), 'Cladding': (12), 'BREEAM Rating': (12)}
                
EXPORT_NAMES = STR_FEATURES + INT_FEATURES + INT_OUTPUTS"""

#lire
#traduire
#enregistrer



"""
This script allows to relay CO2 data between a csv database and a neural network learning script.
It is composed of three steps:
    1. Data importing
    2. Data translation
    3. Data exporting 

Each example in the imported database describes a construction project:
- it provides a series of qualitative and quantitative features expected to impact the CO2 footprints;
- it provides the CO2 footprint values associated to each project, computed externally using LCA evaluation tools.

Qualitative features include :                      #describe categories and list options inside features
    'Sector', 
    'Type', 
    'Basement', 
    'Foundations', 
    'Ground Floor', 
    'Superstructure', 
    'Cladding', 
    'BREEAM Rating'
    
Quantitative features include :
    'GIFA (m2)', 
    'Storeys', 
    'Typical Span (m)', 
    'Typ Qk (kN/m2)'

CO2 footprint values include : 
    'Calculated Total tCO2e', 
    'Calculated tCO2e/m2'

    1. Data importing
    2. Data translation
    3. Data exporting 

"""

"""
    1. Data importing
"""
def open_csv_at_given_line(filename, first_line, delimiter):
    import csv
    reader = csv.reader(open(filename + '.csv', mode='r'), delimiter=delimiter)
    for i in range(first_line):
        reader.__next__()
    header = reader.__next__()
    return header, reader


def build_dict_from_csv(filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)
    CST = dict()
    for f in FEATURES_NAMES:
        CST[f] = []
    for line in reader:
        for f in FEATURES_NAMES:
            index = header.index(f)
            if line[index] not in CST[f]:
                CST[f].append(line[index])
    return CST


CST = build_dict_from_csv('Copy of PM Carbon Data For Release', 5)


# returns X and Y, two lists of size = nb of examples (of data)
# Each element of these lists is a single vector x (or y), a list of ints, values of the first (or last) layer.
def extract_data_from_csv(filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)
    X, Y = [], []
    for line in reader:
        X.append(translate_inputs(header, line))
        Y.append(translate_outputs(header, line))
    return X, Y



def translate_inputs(header, line):
    #importance ordre
    # line[header.index(feature_name)] = ex brown
    x = []
    for feature_name in STR_FEATURES:
        x = x + translate_str_feature(feature_name, line[header.index(feature_name)])
    for feature_name in INT_FEATURES:
        x = x + translate_int_feature(feature_name, line[header.index(feature_name)])
    return x


def translate_outputs(header, line):
    y = []
    for output_name in INT_OUTPUTS:
        y = y + translate_int_feature(output_name, line[header.index(output_name)])
        # line[header.index(output_name) = ex 3200kgCO2e
    return y


def translate_str_feature(feature_name, feature):
    feature_int = CST[feature_name].index(feature)
    return number_to_binary_list(feature_int, len(bin(len(CST[feature_name]) - 1)) - 2)
    # len(bin(len(CST[feature_name]) - 1)) - 2    :
    # - 2 cfr convertir binaire string enleve 0b
    # si tu veux ecrire 10 chiffres, il ne faut que utiliser des chiffres de 0 à 9; 9 = len-1
    # len(CST[feature_name]-1) = nombre d'options > nombre de cases (vides) a fournir


def number_to_binary_list(number, list_size, cut=0):
    number_binary = [int(x) for x in bin(number)[2:]]
    if cut:
        return ([0 for i in range(list_size - len(number_binary))] + number_binary)[:list_size - cut]
    return [0 for i in range(list_size - len(number_binary))] + number_binary


def translate_int_feature(feature_name, feature):
    siz, cut, ampli = INT_PARAMETERS[feature_name]
    feature = int(ampli * string_to_float(feature))
    if not feature:
        return [0 for i in range(siz-cut)]
    if feature >= 2 ** siz:
        return [1 for i in range(siz-cut)]
    return number_to_binary_list(feature, siz, cut)


def translate_binary_list_to_int_value(mylist):
    return sum([(2 ** i) * mylist[len(mylist) - 1 - i] for i in range(len(mylist))])


def translate_binary_output_to_prediction(y):
    pred = dict()
    index = 0
    for output in INT_OUTPUTS:
        siz, cut, ampli = INT_PARAMETERS[output]
        pred[output] = translate_binary_list_to_int_value(y[index:index + siz - cut]) * (2 ** cut) / ampli
        index += siz - cut
    return pred


def string_to_float(string):
    try:
        return float(string.replace(',', '.'))
    except:
        print(string, ": this should be a number")
        return False




def save_translated_data_to_csv(filename, X, Y, delimiter=';'):
    import csv
    CO2_writer = csv.writer(open(filename + '.csv', mode='w'), delimiter=delimiter)

    combined_headers_for_export = []
    for f in STR_FEATURES:
        combined_headers_for_export.append(f)
        for j in range(len(bin(len(CST[f]) - 1)) - 3):
            combined_headers_for_export.append(' ')
    for intlist in (INT_FEATURES, INT_OUTPUTS):
        for f in intlist:
            combined_headers_for_export.append(f)
            for j in range(INT_PARAMETERS[f][0] - INT_PARAMETERS[f][1] - 1):
                combined_headers_for_export.append(' ')
    CO2_writer.writerow([data for data in combined_headers_for_export])

    for i in range(len(X)):
        CO2_writer.writerow(X[i]+Y[i])



def test0():
    X, Y = extract_data_from_csv('Copy of PM Carbon Data For Release', 5)
    print(X[31], Y[31])
    print(X[32], Y[32])
    print(len(bin(max([int(x) for x in CST['GIFA (m2)']]))), 2**16)
    save_translated_data_to_csv('PM_CO2_data_binary', X, Y)


test0()
