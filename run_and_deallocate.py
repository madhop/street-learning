from subprocess import call
import send_mail
call(["python", "street_learning.py"])
call(["az", "vm", "deallocate", "--resource-group", "UmbertoFucci", "--name", "FucciUmberto"])

#send_mail.send()
