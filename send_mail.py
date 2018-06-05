import smtplib

def send():
    gmail_user = 'madhopteam@gmail.com'
    gmail_password = 'dalbahtpower24h'

    sender_name = 'MadHop Team'
    sent_from = gmail_user

    to = ['madhopteam@gmail.com']
    subject = 'Street Learning Notification'
    body = 'Training complete. Check your VM and shut it off otherwise il Matte paga'

    email_text = """\
    From: %s
    To: %s
    Subject: %s

    %s

    """ % (sender_name, ", ".join(to), subject, body)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()
        print('Email sent!')
    except:
        print('Something went wrong...')
