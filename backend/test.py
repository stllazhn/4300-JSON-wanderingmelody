import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

# Set up your credentials and redirect URI
CLIENT_ID = '9db66a8a93514a43baa61db7d3880242'
CLIENT_SECRET = '221735d938054591b54b2feda1074b80'
REDIRECT_URI = 'http://localhost:8888/callback'

# Set up the scope (this will depend on the API features you need)
SCOPE = 'user-library-read'

# Authenticate using OAuth2
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope=SCOPE))

# Now you can make authorized requests
track_id = "11dFghVXANMlKmJXsNCbNl"
audio_features = sp.audio_features([track_id])

print(audio_features)
