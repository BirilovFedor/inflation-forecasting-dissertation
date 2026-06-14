from Data.Dataset import Dataset

Date_columns = ['Date']
Response_columns = ['CPI(Food)', 'CPI(Energy)', 'CPI(Core)']
Explanatory_columns = ['PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5']


def runner():
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'

    dataset = Dataset(filepath, sep=';')

    print(dataset.response['CPI(Food)'].feature)


def main():
    runner()
    '''
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    Data = pd.read_csv(filepath, sep=';')
    Data.info()
    print(Data)
    #print(Data['CPI(Core)'])
    plot_acf(Data['CPI(Core)'])
    plt.show()

    plot_pacf(Data['CPI(Core)'])
    plt.show()
'''
    print("Hello")


if __name__ == '__main__':
    main()


