from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s
import re

#consumer key, consumer secret, access token, access secret.
ckey='asdfghjkfghjk'
csecret='ertyuizxcvbnmcvbn'
atoken='12345-tyunmsdfnmdcvyhni'
asecret='wesdfcvyghzxcvv'
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
