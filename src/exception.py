import sys # manipulate python runtime enviroment
import logging

def error_message_details(error,error_detail:sys): #custome exception handling
    _,_,exc_tb= error_detail.exc_info() 
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in pyhton script nsme [{0}] line number[{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
    return error_message


class CustomException(Exception): # inherit parent exception class
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

