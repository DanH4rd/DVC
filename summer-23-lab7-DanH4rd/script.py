from transformers import pipeline
import praw
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def getPostsInTimeChart(timestamps):
    dates = [datetime.datetime.fromtimestamp(ts).date() for ts in timestamps]
    df = pd.DataFrame( dates, columns=['date']).groupby('date').size().reset_index(name='count')

    fig = px.line(df, x='date', y='count', markers=True)

    # Set the chart title and axes labels
    fig.update_layout(title='Post Occurrences over Time',
                    xaxis_title='Date',
                    yaxis_title='Number of Posts')
    
    # fig.update_layout(xaxis=dict(
    #     type='date',
    #     tickformat= '%b %d, %Y',
    #     dtick= 86400000.0
    # ))


    return fig

def getAverageSentimentChart(preds):
    list_label = ['sadness', 'joy',  'love',  'anger',  'fear', 'surprise']
    values = [0,0,0,0,0,0]
    for pr in preds:
        for i in range(6):
            values[i] = values[i] + pr[i]['score']
    values = [i/len(preds) for i in values]

    return go.Figure(data=[go.Pie(labels=list_label, values=values)])

def getSentimentInTimeBarChart(preds, preds_timestamp):
    list_label = ['sadness', 'joy',  'love',  'anger',  'fear', 'surprise']
    dates = [datetime.datetime.fromtimestamp(ts).date() for ts in preds_timestamp]

    df_sent = pd.DataFrame(preds, columns = list_label).applymap(lambda x: x['score'])
    df_dates = pd.DataFrame( dates[:-2], columns=['date'])

    df_sent_date = pd.concat([df_sent, df_dates], axis = 1)
    df_avg_sent_over_time = (df_sent_date.groupby('date').sum() / df_sent_date.groupby('date').count()).reset_index()
    x = df_avg_sent_over_time['date']

    fig = go.Figure(go.Bar(x=x, y=df_avg_sent_over_time['sadness'], name='Sadness'))
    fig.add_trace((go.Bar(x=x, y=df_avg_sent_over_time['joy'], name='Joy')))
    fig.add_trace((go.Bar(x=x, y=df_avg_sent_over_time['love'], name='Love')))
    fig.add_trace((go.Bar(x=x, y=df_avg_sent_over_time['anger'], name='Anger')))
    fig.add_trace((go.Bar(x=x, y=df_avg_sent_over_time['fear'], name='Fear')))
    fig.add_trace((go.Bar(x=x, y=df_avg_sent_over_time['surprise'], name='Surprise')))

    fig.update_layout(barmode='stack')
    fig.update_xaxes(categoryorder='total ascending')
    # fig.update_layout(xaxis=dict(
    #         type='date',
    #         tickformat= '%b %d, %Y',
    #         dtick= 86400000.0
    #     ))
    
    return fig

def getSentimentInTimeLineChart(preds, preds_timestamp):
    list_label = ['sadness', 'joy',  'love',  'anger',  'fear', 'surprise']
    dates = [datetime.datetime.fromtimestamp(ts).date() for ts in preds_timestamp]

    df_sent = pd.DataFrame(preds, columns = list_label).applymap(lambda x: x['score'])
    df_dates = pd.DataFrame( dates[:-2], columns=['date'])

    df_sent_date = pd.concat([df_sent, df_dates], axis = 1)
    df_avg_sent_over_time = (df_sent_date.groupby('date').sum() / df_sent_date.groupby('date').count()).reset_index()
    x = df_avg_sent_over_time['date']

    fig = go.Figure(go.Scatter(x=x, y=df_avg_sent_over_time['sadness'], name='Sadness'))
    fig.add_trace((go.Scatter(x=x, y=df_avg_sent_over_time['joy'], name='Joy')))
    fig.add_trace((go.Scatter(x=x, y=df_avg_sent_over_time['love'], name='Love')))
    fig.add_trace((go.Scatter(x=x, y=df_avg_sent_over_time['anger'], name='Anger')))
    fig.add_trace((go.Scatter(x=x, y=df_avg_sent_over_time['fear'], name='Fear')))
    fig.add_trace((go.Scatter(x=x, y=df_avg_sent_over_time['surprise'], name='Surprise')))

    fig.update_layout(title='Sentiment change over Time',
                xaxis_title='Date',
                yaxis_title='Sentiment avg value')

    # fig.update_layout(xaxis=dict(
    #         type='date',
    #         tickformat= '%b %d, %Y',
    #         dtick= 86400000.0
    #     ))
    
    return fig

def getCharactersOverTime(posts, timestamps):
    dates = [datetime.datetime.fromtimestamp(ts).date() for ts in timestamps]

    df_text_len = pd.DataFrame([len(p) for p in posts], columns = ['text_len'])
    df_dates = pd.DataFrame( dates[:-2], columns=['date'])

    df_text_len_date = pd.concat([df_text_len, df_dates], axis = 1)
    df_avg_sent_over_time = df_text_len_date.groupby('date').sum().reset_index()

    fig = px.line(df_avg_sent_over_time, x='date', y='text_len', markers=True)

    # Set the chart title and axes labels
    fig.update_layout(title='Characters written over Time',
                    xaxis_title='Date',
                    yaxis_title='Number of Characters')
    
    # fig.update_layout(xaxis=dict(
    #     type='date',
    #     tickformat= '%b %d, %Y',
    #     dtick= 86400000.0
    # ))

    return fig

def getAvgScoreOverTime(scores, timestamps):
    dates = [datetime.datetime.fromtimestamp(ts).date() for ts in timestamps]

    df_score = pd.DataFrame(scores, columns = ['score'])
    df_dates = pd.DataFrame( dates[:-2], columns=['date'])

    df_score_date = pd.concat([df_score, df_dates], axis = 1)
    df_avg_score_over_time = (df_score_date.groupby('date').sum() / df_score_date.groupby('date').count()).reset_index()

    fig = px.line(df_avg_score_over_time, x='date', y='score', markers=True)

    # Set the chart title and axes labels
    fig.update_layout(title='Average score over Time',
                    xaxis_title='Date',
                    yaxis_title='Avg score')
    
    # fig.update_layout(xaxis=dict(
    #     type='date',
    #     tickformat= '%b %d, %Y',
    #     dtick= 86400000.0
    # ))

    return fig

@st.cache_data
def GetSubredditPosts(subreddit_name, limit):
    # create a Reddit instance using PRAW
    reddit = praw.Reddit(client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent)


    # get the subreddit object
    subreddit = reddit.subreddit(subreddit_name)

    # get the top 10 text posts from the past week
    top_posts = subreddit.new(limit=limit)

    posts = []
    time_stamps = []
    scores = []

    # loop through each post and print its title, author, and creation time
    for post in top_posts:
        if post.is_self: # only consider text posts, not links
            posts.append(post.selftext)
            time_stamps.append(post.created_utc)
            scores.append(post.score )

    return(posts, time_stamps, scores)

def GetPredictions(str_list, time_stamps, progress_bar, scores):
    classifier = pipeline("text-classification", model='bhadresh-savani/albert-base-v2-emotion', return_all_scores=True)
    predictions = []
    pred_posts = []
    predictions_time_stamps = []
    predictions_scores = []
    for i in range(len(str_list)):
        try:
            predictions = predictions +  classifier(str_list[i])
            predictions_time_stamps.append(time_stamps[i])
            predictions_scores.append(scores[i])
            pred_posts.append(str_list[i])
        except:
            pass

        progress_bar.progress(int(100 * (i+1) / len(str_list)))

    return (pred_posts, predictions, predictions_time_stamps, predictions_scores)

##############################################

st.title('Enter following Reddit app credentials')
client_id       = st.text_input("client_id")
client_secret   = st.text_input("client_secret")
user_agent      = st.text_input("user_agent")

st.title('Reddit subreddit analysis')

# specify the subreddit you want to scrape
subreddit_name = st.text_input("Enter subreddit name")
post_limit = st.number_input('Insert maximum number of posts to scrap', step  = 1, min_value = 1, max_value=1000, value = 100)

if st.button("Run analysis"):

    # subreddit_name = "learnpython"
    st.write("Sentiment analysis progress")

    progress_bar_sent = st.progress(0, text = 'Scrapping posts, when finished analysis will start')

    posts, time_stamps, scores = GetSubredditPosts(subreddit_name, post_limit)

    classifier = pipeline("text-classification", model='bhadresh-savani/albert-base-v2-emotion', return_all_scores=True)

    pred_posts, predictions, pred_time_stamps, pred_scores  = GetPredictions(posts, time_stamps, progress_bar_sent, scores)
    st.write('Out of ', len(posts), 'scrapped posts', len(pred_posts), 'were succesfully analysed')
    st.write(str(round(len(pred_posts)/len(posts) * 100,2)), '\% successs rate.')

    st.title('Posts over time')
    st.write("Posts for which sentiment analysis failed are not included")
    st.plotly_chart(getPostsInTimeChart(pred_time_stamps))

    st.title('Characters written over time')
    st.write("Posts for which sentiment analysis failed are not included")
    st.plotly_chart(getCharactersOverTime(pred_posts, pred_time_stamps))

    st.title('Average sentiment')
    st.write("Posts for which sentiment analysis failed are not included")
    st.plotly_chart(getAverageSentimentChart(predictions))

    st.title('Average sentiment over time (bar)')
    st.write("Posts for which sentiment analysis failed are not included")
    st.plotly_chart(getSentimentInTimeBarChart(predictions, pred_time_stamps))

    st.title('Average sentiment over time (line)')
    st.write("Posts for which sentiment analysis failed are not included")
    st.plotly_chart(getSentimentInTimeLineChart(predictions, pred_time_stamps))

    st.title('Average score over time')
    st.write("Posts for which sentiment analysis failed are not included")
    st.plotly_chart(getAvgScoreOverTime(pred_scores, pred_time_stamps))

