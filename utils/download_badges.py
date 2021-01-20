import requests
def download_image(url, output_image_file):
    r = requests.get(url)
    with open(output_image_file, 'w') as f:
        if isinstance(r.content, bytes):
            content = r.content.decode()
        else:
            content = r.content
        f.write(content)

if __name__=="__main__":
    
    download_image("https://static.pepy.tech/personalized-badge/vectorhub-nightly?period=total&units=none&left_color=black&right_color=purple&left_text=Total%20Downloads", 
    "assets/total_downloads.svg")
    download_image("https://static.pepy.tech/personalized-badge/vectorhub-nightly?period=week&units=none&left_color=black&right_color=purple&left_text=Weekly%20Downloads",
    "assets/weekly_downloads.svg")
    download_image("https://static.pepy.tech/personalized-badge/vectorhub-nightly?period=month&units=none&left_color=black&right_color=purple&left_text=Monthly%20Downloads",
    "assets/monthly_downloads.svg")
