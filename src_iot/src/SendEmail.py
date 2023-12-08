import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def create_table_html(data):
    table_rows = ""
    for row in data:
        table_cells = ""
        for cell in row:
            table_cells += f"<td>{cell}</td>"
        table_rows += f"<tr>{table_cells}</tr>"

    table_html = f"""
    <table border="1" cellpadding="5" cellspacing="0">
        {table_rows}
    </table>
    """

    return table_html



def send_email(table_data):
    
    
    # Create the HTML table
    html_table = create_table_html(table_data)

    # Email details
    sender_email = "votiendat08112001@gmail.com"
    receiver_email = "n19dcat016@student.ptithcm.edu.vn"
    subject = "Các cuộc tấn công phân tích từ gói tin trong mạng"
    password = "owpiaogxnyhoupkw"

    # Compose the email
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email


    html_content = MIMEText(html_table, "html")

    message.attach(html_content)

    # Send the email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())