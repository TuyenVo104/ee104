Name: Jose Ramirez Estrada
Class: EE 104

Reference: https://www.twilio.com/docs/sms/quickstart/python
Reference: https://stackoverflow.com/questions/59631659/i-dont-understand-why-i-get-a-too-many-values-to-unpack-error

About this code:

This code is composed of the GUI login from simplifiedpython.net and the Simple Registartion
Form from geeksforgeeks.org (Refernces above). The file laso contains code frm the email code
that at provided from ee 104 at sjsu belonging t0 professor Chritopher Pham.

# Module 1 GUI REGISTRATION DATABSE (Description of Module 1) SKIP to module 2 below!
# Module 2 Infor listeed below
In order to operate both programs from the same source code each source code for each 
program had to be chnaged in some manner. For example, grid and packs cannot be 
used together in the same code so the code protion from the GUI login
program had to be changed to work with grid. Additionally, the registration form program code had to be
changed to operate as a "pop up screen" upon user login within the GUI code. To make the Registration
form program operate with the GUI interface global vaeiables had to be created for each user input and
related variables. Other program additions include background imagen, which was implimented thanks
to the stackoverflow questions page (referenced above). Other changes include atomatic closure of
windows upon completion of tasks, such as registration form submission. Other chaages also
include popups that notify the user of submission form success and empty text boxes. Other changes include
changing the username and password files from a ".txt" to a "xlsx" file. 


# Module 2 VACCINATION 
The code was merged with module 1's login and registration form.
The code that twas provided in class "sendMailFromCSV.py" was used and modified
to work with a csv datfarame that was managed with the pandas module (Pandas must be installed)
(To utilize the email server smtlib and ssl were imported)
User data is saved to SimpleRegistrationDatabase.xlsx and COVID_Database.csv
date, datetime and timedelta were used to manipualte the date inputs.
(datetime must be installed and date, datetime and time delta must be imported)
One main error that was encountered was a datime error that involved date formating
to fix this issue a timedelta of 5 minutes as added to the original forst vacination date input.
Ofcourse a timedelta of 21 days was also added for the second vaccination date.
The text messaging feature utlizes twilio's messaging sms messaging service
(Reference: https://www.twilio.com/docs/sms/quickstart/python)
the input for the phonenumber must be a 10digit phone number without parnthesis ot dashes
for example 8312224456.
(Install twilio module) (pip install twilio)


***MAKE SURE THAT THE FOLLOWING PYTHON MODULES ARE INSTALLED***

from tkinter import *
from openpyxl import *
from datetime import date, datetime, timedelta (pip install datetime)
import pandas as pd (pip install pandas)
import os
import smtplib, ssl
from twilio.rest import Client (pip install twilio)



Instructions: 
- make sure that the modules above are installed 
- run applications
- sign in or register
- click OK to exit! in the popup windiow to close login screen (unecessary screen after loging in)
- Register or input information
  - The phone number must be in a 10 digit format without dashes or parentheis ex. 0123456789
  - enter email in regular email format ex. josedogsnop@gmail.com
  - enter date in the follwoing format with dashes as shown (YYYY-MM-DD) ex. 2021-03-02
  - submit
  - proffesor toll to check a date will pop up. the original date that was inputed for the first
    vaccine and the secon vaccine date are shown to the left in the black box in red letters for refrernce.
  - Enter a date in the same format that i sthree days prior to the seconf vaccination date (YYYY-MM_DD)
  - click OK to exit to close the date check screen
- registartion form clears and is ready for a new registration


 
- 






 
             
