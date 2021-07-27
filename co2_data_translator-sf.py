STR_FEATURES = ['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding', 'BREEAM Rating']
INT_FEATURES = ['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN/m2)']
FEATURES_NAMES = STR_FEATURES + INT_FEATURES
INT_OUTPUTS = ['Calculated Total tCO2e', 'Calculated tCO2e/m2']
INT_PARAMETERS = {'GIFA (m2)': (16, 8, 1), 'Storeys': (5, 1, 1), 'Typical Span (m)': (7, 0, 5), 'Typ Qk (kN/m2)': (5, 0, 2),
                  'Calculated Total tCO2e': (15, 7, 1), 'Calculated tCO2e/m2': (11, 4, 1000)} #(siz,cut, precision)
"""
STR_PARAMETERS = {'Sector': (7), 'Type': (5), 'Basement': (4), 'Foundations': (7),
                  'Ground Floor': (6), 'Superstructure': (12), 'Cladding': (12), 'BREEAM Rating': (12)}
                
EXPORT_NAMES = STR_FEATURES + INT_FEATURES + INT_OUTPUTS"""

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
    Sector ['Other', 'Residential', 'Cultural', 'Educational', 'Mixed Use', 'Commercial', 'Industrial']
    Type ['New Build (Brownfield)', 'New Build (Greenfield)', 'Mixed New Build/Refurb']
    Basement ['None', 'Partial Footprint', 'Full Footprint']
    Foundations ['Piled Ground Beams', 'Mass Pads/Strips', 'Raft', 'Piles (Pile Caps)', 'Reinforced Pads/Strips', '']
    Ground Floor ['Suspended RC', 'Ground Bearing RC', 'Suspended Precast', 'Raft', 'Other']
    Superstructure ['In-Situ RC', 'CLT Frame', 'Steel Frame, Precast', 'Masonry, Concrete', 'Steel Frame, Composite', 'Steel Frame, Other', 'Masonry, Timber', 'Other', 'Timber Frame', 'Steel Frame, Timber']
    Cladding ['Masonry + SFS', 'Lightweight Only', 'Stone + Masonry', 'Glazed/Curtain Wall', 'Masonry Only', 'Stone + SFS', 'Other', 'Lightweight + SFS', 'Timber + SFS', 'Timber Only']
    BREEAM Rating ['Unknown', 'Very Good', 'Excellent', 'Good', 'Passivhaus', 'Outstanding']
    
Quantitative features include :
    'GIFA (m2)', 
    'Storeys', 
    'Typical Span (m)', 
    'Typ Qk (kN/m2)'

CO2 footprint values include : 
    'Calculated Total tCO2e', 
    'Calculated tCO2e/m2'

"""

"""
    1. Data importing
    
    The first step of the data handling consists in importing data from a csv file into the script. 
    Features imported are split in string features and integer features, depending on their qualitative or quantitative nature.

    Input : csv file
    Output : 
    - CST : dictionary of features and associated possible values;
    - X : matrix assembling the feature description of each example in the training set, input matrix for the neural network, 
            matrix size : (number of examples x number of features) 
    - Y : matrix assembling the CO2 values of each example in the training set, output matrix for the neural network.
            matrix size : (number of examples x number of CO2 values) 
            
    Uses :
    - open_csv_at_given_line
    - build_dict_from_csv
    - extract_data_from_csv
    
    This step is done simultaneously for all examples (X or Y).
    It calls the embedded translation function to translate each data one by one.
    
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


def extract_data_from_csv(filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)
    X, Y = [], []
    for line in reader:
        X.append(translate_inputs(header, line))
        Y.append(translate_outputs(header, line))
    return X, Y

"""
    2. Data translation
    
    The second step of the data handling consists in translating data between decimal or text format and binary format.
    Translating to binary format is necessary for the data to be compatible and read by the neural network.
    Inversely, translating to text/decimal format is necessary for the data to be readable/interpretable by the user.
    
    Converting to binary format is achieved :
    - for text values (string features), by associating each possible value (discrete number) to one binary number, 
        in this case, the index of the value in the feature list ; 
    - for decimal values (int features) by associating a range of possible values (continuous number) to a binary number;
        The discretization of the decimal values is obtained for each feature, by a defined size, cut and precision:
        
         -Size : bit count (number) needed to code the upper bound of the feature. 
         The upper bound of the feature can either be the highest decimal value in the training set, 
         or a higher decimal number identified as the maximal value that this feature could take.
         -Precision : coefficient to convert float decimals into integers whilst maintaining required data precision
         When translating decimals into binary, numbers behind the comma are lost. 
         A preliminary multiplication of decimal numbers by a precision factor avoids loosing this information.
         The precision is the inverse of the range of decimal units identified as necessary for good CO2 prediction.
         -Cut : bit count (order of magnitude) used for the rounding up of the binary data.         
         Data values can be rounded up to a certain extent without affecting the validity of CO2 prediction. 
         The cut value is the number of bits that can be removed from the end of the binary representation of data.
         Cut is usually associated to the size (bit count) of the discretization.
         
    Optimal parameter values ensure holistic (size) and sufficient (precision) data to achieve CO2 prediction, 
    without saturating storage space with unneccessary information (cut). 
    Initial values can result from common sense and general knowledge (human learning), 
    optimal values can result from machine learning.
         
    --------------------------------------------------------------------------------------------------------------------     
         
    Selected parameters: 
    
        'GIFA (m2)',
        Size <decimal> : 65 536 <m2>
        Size <binary> : 16
        Precision <decimal> : 1
        Cut <decimal> : 256 <m2>
        Cut <binary> : 8    
     
        'Storeys', 
        Size <decimal> : 32 
        Size <binary> : 5
        Precision <decimal> : 1
        Cut <decimal> : 2
        Cut <binary> : 1       
        
        'Typical Span (m)', 
        Size <decimal> : 128  <25.6m*5>
        Size <binary> : 7
        Precision <decimal> : 5 <20cm>
        Cut <decimal> : 0
        Cut <binary> : 0  
                
        'Typ Qk (kN/m2)'
        Size <decimal> : 32  <16m*2>
        Size <binary> : 5
        Precision <decimal> : 2 <50cm>
        Cut <decimal> : 0
        Cut <binary> : 0      

        'Calculated Total tCO2e', 
        Size <decimal> : 32768 <kgCO2e>
        Size <binary> : 15
        Precision <decimal> : 1
        Cut <decimal> : 128 <kgCO2e>
        Cut <binary> : 7          
                
        'Calculated tCO2e/m2'
        Size <decimal> : 2048 <2.048kgCO2e/m2*1000>
        Size <binary> : 11
        Precision <decimal> : 1000 
        Cut <decimal> : 16 <kgCO2e/m2>
        Cut <binary> : 4  

    --------------------------------------------------------------------------------------------------------------------         

    Input : In decimal or String
    - x : vector containing the feature description of each example in the training set, input vectors for the neural network, 
            vector size : (number of features x 1) 
    - y : matrix assembling the CO2 values of each example in the training set, output vectors for the neural network.
            vector size : (number of CO2 values x 1) 
    Output : In binary
                
    Uses :
    
    - translate_inputs
    - translate_outputs
    - translate_str_feature
    - number_to_binary_list
    - translate_int_feature
    - translate_binary_list_to_int_value
    - translate_binary_output_to_prediction
    - string_to_float
    
    This step is done individually for on example at a time (x or y)...

"""

def translate_inputs(header, line):
    """
    Comments :
    - line[header.index(feature_name)] = used to query a feature_value

    --------------------------------------------------------------------------------------------------------------------

    input : feature name and training example line
    output : translated line

    """

    x = []
    for feature_name in STR_FEATURES:
        x = x + translate_str_feature(feature_name, line[header.index(feature_name)])
    for feature_name in INT_FEATURES:
        x = x + translate_int_feature(feature_name, line[header.index(feature_name)])
    return x


def translate_outputs(header, line):
    """

    input : feature name and training example line
    output : translated output line

    """

    y = []
    for output_name in INT_OUTPUTS:
        y = y + translate_int_feature(output_name, line[header.index(output_name)])
        # line[header.index(output_name) = ex 3200kgCO2e
    return y


def number_to_binary_list(number, list_size, cut=0):
    """
    input : int/float number (decimal), int feature size (binary), int feature cut (binary)
    output : number [bit list] translated to binary and formatted version (size, cut, precision)
    """

    number_binary = [int(x) for x in bin(number)[2:]]
    if cut:
        return ([0 for i in range(list_size - len(number_binary))] + number_binary)[:list_size - cut]
    return [0 for i in range(list_size - len(number_binary))] + number_binary

def translate_str_feature(feature_name, feature_value):
    """
    Comments :

    Translating to binary language
    - "len-1" : to code numbers from 1 to 10 (len), you only need digits from 0 to 9 (len-1)
    - "len(CST[feature_name])" : number of different feature values fot this feature name
    - "len-2" : conversion into binary using bin() returns a string of numbers starting with "0b", to be removed

    Translating text strings
    - text features are translated in a set/fixed order, and associated to their corresponding index, order counts!

    --------------------------------------------------------------------------------------------------------------------
    input : feature_name (string), feature_value (string)
    output : feature_value [bit list] translated to binary and formatted version (size, cut, precision)
    """

    feature_int = CST[feature_name].index(feature_value)
    return number_to_binary_list(feature_int, len(bin(len(CST[feature_name]) - 1)) - 2)

def string_to_float(string):
    """
    input : decimal number in string with "," for decimal separation
    output : decimal number in float with "." for decimal separation
    """

    try:
        return float(string.replace(',', '.'))
    except:
        print(string, ": this should be a number")
        return False

def translate_int_feature(feature_name, feature_value):
    """
    Translates int feature values to set binary format (size, cut, precision),
    - if feature value is too small  -> returns a list of 0s, with good feature format
        here default inferior threshold is 0
    - if feature value is bigger than max feature value  -> returns a list of 1s, with good feature format

    --------------------------------------------------------------------------------------------------------------------
    Comment :
    if not feature_value ; with type(feature_value) = int <=> if feature_value == 0
    dans notre cas ou f_val est positif, => if feature_value <= 0

    --------------------------------------------------------------------------------------------------------------------
    input : feature_name (string), feature_value (string)
    output : feature_value [bit list] translated to binary and formatted version (size, cut, precision)
    """

    siz, cut, ampli = INT_PARAMETERS[feature_name]
    feature_value = int(ampli * string_to_float(feature_value))
    if not feature_value:
        return [0 for i in range(siz-cut)]
    if feature_value >= 2 ** siz:
        return [1 for i in range(siz-cut)]
    return number_to_binary_list(feature_value, siz, cut)

def translate_binary_list_to_int_value(mylist):
    """
    input : binary number as list
    output : int number raised to set precision (result = decimal value * precision)
    """
    return sum([(2 ** i) * mylist[len(mylist) - 1 - i] for i in range(len(mylist))])

def translate_binary_output_to_prediction(y):
    """
    input : predicted value as binary number
    output : predicted value as decimal number
    """

    pred = dict()
    index = 0
    for output in INT_OUTPUTS:
        siz, cut, ampli = INT_PARAMETERS[output]
        pred[output] = translate_binary_list_to_int_value(y[index:index + siz - cut]) * (2 ** cut) / ampli
        index += siz - cut
    return pred


"""
    3. Data exporting 
    
    The third step of the data handling consists in exporting data from the script to a csv file.
    This data can then be used for ...
    -Learning ...
    
    Input : 
    - X : matrix assembling the feature description of each example in the training set, 
            input matrix for the neural network, in binary format 
            matrix size : (number of examples x number of features) 
    - Y : matrix assembling the CO2 values of each example in the training set, 
            output matrix for the neural network, in binary format 
            matrix size : (number of examples x number of CO2 values) 
            
    Output : 
    csv file describing data in binary format, one column per bit value

    Uses :
    -save_translated_data_to_csv

    This step is done simultaneously for all examples (X or Y).    
    
"""

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

"""
    4. Testing 
"""

CST = build_dict_from_csv('Copy of PM Carbon Data For Release', 5)

def test0():
    X, Y = extract_data_from_csv('Copy of PM Carbon Data For Release', 5)
    print(X[31], Y[31])
    print(X[32], Y[32])
    print(len(bin(max([int(x) for x in CST['GIFA (m2)']]))), 2**16)
    save_translated_data_to_csv('PM_CO2_data_binary', X, Y)


#test0()

for k in CST:
    print(k, CST[k])

a = int(1)
print(bool(a))
a -= 1
print(bool(a))
a -= 1
print(bool(a))