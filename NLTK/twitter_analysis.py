from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s
import re

#consumer key, consumer secret, access token, access secret.
ckey='uUMqlnxxE8T49Nzlv7mgwONZr'
csecret='JND0fpbncSxrRxs14SwMCdDREozZO40FTI81GPGPygU5ZTkcV6'
atoken='997005544886956034-xfqFDRJajJeCSoWJzyGn6dXiyuSfv7c'
asecret='gqfZITF7M4BTPeFuj0jlyIqa59TmxYHVRYg3OlGvtBKae'
class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = ascii(all_data['text'])
        tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)
        print('\n')
        if confidence*100 >= 80:
            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        return True
        
    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])