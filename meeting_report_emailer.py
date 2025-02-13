import os
import json
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from typing import Dict, List
from pathlib import Path
import logging
from dotenv import load_dotenv
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Email configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')

class MeetingReportEmailer:
    def __init__(self, meeting_logs_dir: str = "meeting_logs"):
        self.meeting_logs_dir = Path(meeting_logs_dir)
        
    def parse_meeting_log(self, log_path: Path) -> Dict:
        """Parse a meeting log file into a structured format"""
        meeting_data = {
            'meeting_info': {},
            'questions_and_responses': []
        }
        
        current_section = None
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('==='):
                    current_section = line.strip('= ')
                    continue
                    
                if ':' in line:
                    key, value = map(str.strip, line.split(':', 1))
                    if current_section == 'Meeting Summary':
                        meeting_data['meeting_info'][key.lower()] = value
                    elif current_section == 'Questions and Responses':
                        if key.startswith('Q'):
                            meeting_data['questions_and_responses'].append({'question': value})
                        elif key.startswith('A') and meeting_data['questions_and_responses']:
                            meeting_data['questions_and_responses'][-1]['answer'] = value
                            
        return meeting_data

    def generate_html_content(self, meeting_data: Dict) -> str:
        """Generate HTML content with embedded CSS for the email"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px 5px 0 0;
                    margin-bottom: 20px;
                }
                .meeting-info {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .qa-section {
                    margin-bottom: 30px;
                }
                .question {
                    background-color: #e3f2fd;
                    padding: 10px 15px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }
                .answer {
                    background-color: #f5f5f5;
                    padding: 10px 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .info-item {
                    margin-bottom: 10px;
                }
            </style>
        </head>
        <body>
        """
        
        # Add header
        html_content += f"""
            <div class="header">
                <h1>Metaverse911-Meeting Summary Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add meeting information
        html_content += """
            <div class="meeting-info">
                <h2>Meeting Details</h2>
        """
        
        for key, value in meeting_data['meeting_info'].items():
            html_content += f'<div class="info-item"><strong>{key.title()}:</strong> {value}</div>'
        
        html_content += "</div>"
        
        # Add Q&A section
        html_content += """
            <div class="qa-section">
                <h2>Discussion Details</h2>
        """
        
        for i, qa in enumerate(meeting_data['questions_and_responses'], 1):
            html_content += f"""
                <div class="question">
                    <strong>Q{i}:</strong> {qa['question']}
                </div>
                <div class="answer">
                    <strong>A{i}:</strong> {qa['answer']}
                </div>
            """
        
        html_content += """
            </div>
            </body>
            </html>
        """
        
        return html_content

    def send_email(self, recipient_email: str, html_content: str, subject: str = "Meeting Summary Report"):
        """Send the HTML email"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = SENDER_EMAIL
            msg['To'] = recipient_email

            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
                
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False


    def process_and_send_report(self, log_file: str) -> bool:
        """Process a meeting log file and send the report"""
        try:
            log_path = Path(log_file)
            if not log_path.exists():
                logger.error(f"Log file not found: {log_path}")
                return False

            # Read the log file content
            with open(log_path, 'r') as file:
                log_content = file.read()

            # Use regex to extract the recipient email (expects a line like "Email: example@domain.com")
            email_match = re.search(r'Email:\s*([\w\.-]+@[\w\.-]+)', log_content)
            if email_match:
                recipient_email = email_match.group(1)
            else:
                logger.error("Recipient email not found in log file.")
                return False

            # Parse the meeting log and generate HTML content
            meeting_data = self.parse_meeting_log(log_path)
            html_content = self.generate_html_content(meeting_data)

            # Send the email report
            return self.send_email(
                recipient_email,
                html_content,
                f"Meeting Summary - {meeting_data['meeting_info'].get('date', 'Undated')}"
            )

        except Exception as e:
            logger.error(f"Error processing meeting report: {str(e)}")
            return False



