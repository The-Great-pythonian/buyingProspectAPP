import numpy
import streamlit
import pickle
import pandas as pd

loaded_model = pickle.load(open('trained_pipeModel.sav','rb'))
#create function to handle predicition

def FMG_prediction (user_input_data):
    #convert to array
    Input_array = numpy.asarray(user_input_data)
    Input_array_reshaped = Input_array.reshape(1, -1)
    make_prediction = loaded_model.predict(Input_array_reshaped)
    # percentage calulation of make prediction relative to other items
    path = r'c:\Users\INDOMITABLE ROCK\PycharmProjects\file_folder\sales_nonull.csv'
    df = pd.read_csv(path)

    a = make_prediction - numpy.percentile(df['BuyingProspect'], 0)
    #we remove the outliers to get realistic percentage so we use 98%
    b = numpy.percentile(df['BuyingProspect'],100) - numpy.percentile(df['BuyingProspect'], 0)
    perc = a / b * 100
    # #presentation of the result in readable format
    perc= str(perc).strip('[]')
    perc = round(float(perc),2)
    make_prediction = str(make_prediction).strip('[]')
    # make_prediction = float(make_prediction)
    return f'Buying prospect rating is {make_prediction}.  The percentage that customer would buy on Jumia is {perc}%'

#construct interface for user data input
def main():
    #give a title
    streamlit.title(' Find item to sell on this Mall ?  ')
    streamlit.title('ML finds its buying prospect  to increase selling on the Mall')
    # get input data from user
    path = r'c:\Users\INDOMITABLE ROCK\PycharmProjects\file_folder\sales_nonull.csv'
    df = pd.read_csv(path)
    #create a drop down menu from csv file
    #selected_items = streamlit.selectbox("Select Category", df['items'])
    #for easy search will shall apend item id and Items_Category
    selected_items = streamlit.selectbox("Select similar product you want to sell on jumia", df['items']+
                                         "," + "Items_Category:"
                                         +df['Items_Category'].astype(str) + "," +"items_id:" +df['Items_id'].astype(str))
    #select an item
    filtered_df = df[df['items']+ "," + "Items_Category:" +
                     df['Items_Category'].astype(str) + ","
                     +"items_id:" +df['Items_id'].astype(str) == selected_items]

    # #Pass the values of  filtered_df to variables,
    Items_id =  filtered_df['Items_id']  #produce only items id from  filtered_df

    Items_display = streamlit.write(Items_id)  #string
    Items_Category =  filtered_df['Items_Category'] ##produce only Items_Category from  filtered_df
    Items_Category_display = streamlit.write(Items_Category)#string


    Price = streamlit.number_input('Enter a price to optmise buying .g 499.99 ', min_value=0.99, max_value=4999.99, value=0.99, step=1.00, key='Price')
    Discount = streamlit.number_input('Enter discount',min_value=0, max_value=1999, value=0, step=1, key='Discount')
    # code for prediction
    detection = ""  # declare this variable to hold result like empty list
    # mylist = []
    if streamlit.button('click here for Buying prospect of this item'):
        detection = FMG_prediction ([int(Items_id),int(Items_Category),Price,Discount])
        # convert inputs into a single parameter using list [1,2]
        # FMG_prediction ...call the function to process input
    streamlit.success(detection)


if __name__ == '__main__':
    main()

# web app  on your desktop local host
#run  on your pycharm terminal ' streamlit run sellingProspectAPP.py '
# if using  command window ensure the path is correct..  c:\users\idom...\pycharm..\machin learing>
# that is pionting to your python file
