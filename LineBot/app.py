# -*- coding: UTF-8 -*-
import os
import threading
import time
import open
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
richmenuid = "richmenu-d7113816ddd71e98847932785c0bbc3a"
import example
someoneId = "U1947a39afd2d1d33234855411c3f6584"
app = Flask(__name__)
# Channel Access Token
# line_bot_api = LineBotApi('jj791irCY1aUhYeIFWibOn2/g1TdAB5y14ABHFuFvmFn3mdtZgeS/ph5UoPvYfnKw2vX5A0d1amXhNLuETbmWWTtIKGT7zoi/J/RdnJ1tl0O8Az4U4Bx9qGegTgE30eKfOw1lUeJJRz7qZFZjwSgbAdB04t89/1O/w1cDnyilFU=')
line_bot_api = LineBotApi('ax8OThzkpOiuMr4BT6w+X0kgcV2rjyUXuiy3/1yyB0E71ulCDOOBMxT2CjFqLsWBl5iqnLaPb7WZE5YUTVDzzb6wiMDGaDfYo6oF7duBmDH9NBu3JL/EsI1Jj5Zw4pP95e6Znf4Gp9Mp6y5cK62kAQdB04t89/1O/w1cDnyilFU=')
# Channel Secret
#handler = WebhookHandler('3b9b043a7aa13ba058929ce327dbe6f4')
handler = WebhookHandler('0f3018957e5d8b791b682b51ed9a412a')

# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 處理訊息 

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text.lower()
    userId = event.source.user_id
    # id = event.source
    # print(id)
    if msg == 'info':
        message = TemplateSendMessage(
        alt_text='Buttons template',
        template=ButtonsTemplate(
        thumbnail_image_url='https://live.staticflickr.com/3894/14165550960_89a2ea53cc_b.jpg',
        title='風險管家協助您下單並提醒您此單違約率喔',
        text='輕觸"USER GUIDE"來下單吧',
        actions=[
            URITemplateAction(
                label='Fintech by Roger',
                uri='http://mirlab.org/jang/courses/finTech/'
            ),
            URITemplateAction(
                label='Yahoo stock',
                uri='https://tw.stock.yahoo.com'
            ),
            URITemplateAction(
                label='Our HackMD',
                uri='https://hackmd.io/GEOwwSTrRUSFN1RaCVqRcA?view'
            ),

            ]
            )
        )

        line_bot_api.reply_message(event.reply_token, message)
    elif msg == "developers":

        showDevelopers(userId, event)
        message = TemplateSendMessage(
        alt_text='Buttons template',
        template=ButtonsTemplate(
        thumbnail_image_url='https://live.staticflickr.com/3894/14165550960_89a2ea53cc_b.jpg',
        title='All developers',
        text='Instructors: Ann and Roger\nLineBot Developer:r09525126',
        actions=[
            URITemplateAction(
                label='高為勳',
                uri='https://www.facebook.com/wilson.kao.112'
            ),
            URITemplateAction(
                label='李亭臻',
                uri='https://www.facebook.com/saintpaulstammy'
            ),
            URITemplateAction(
                label='Ying-hsuan Chen',
                uri='https://www.facebook.com/profile.php?id=100009206584411'
            ),
            URITemplateAction(
                label='Winnie Chang',
                uri='https://www.facebook.com/profile.php?id=100000136617830'
            ),
            ]
            )
        )
        #line_bot_api.reply_message(event.reply_token, message)

    elif msg == "transaction":
        #print()
        res = str(example.main())
        message = TextSendMessage(text=res)


        ButtonsTemplateM = TemplateSendMessage(
                            alt_text='Buttons template',
                            template=ButtonsTemplate(
                                title='開始買賣囉',
                                text='請選擇您想使用的ID',
                                actions=[
                                    PostbackTemplateAction(
                                        label='ID:1',
                                        data="1&1"
                                    ),
                                    PostbackTemplateAction(
                                        label='ID:2',
                                        data='1&2'
                                    ),
                                    PostbackTemplateAction(
                                        label='ID:3',
                                        data='1&3'
                                    ),
                                    PostbackTemplateAction(
                                        label='ID:4',
                                        data='1&4'
                                    ),

                                ]
                            )
        )

        line_bot_api.reply_message(event.reply_token, ButtonsTemplateM)


    else:
        emoji = [
            {
                "index": 0,
                "productId": "5ac21a18040ab15980c9b43e",
                "emojiId": "087"
            },
            {
                "index": 12,
                "productId": "5ac21a18040ab15980c9b43e",
                "emojiId": "007"
            },
            {
                "index": 19,
                "productId": "5ac21a18040ab15980c9b43e",
                "emojiId": "007"
            },
            {
                "index": 32,
                "productId": "5ac21a18040ab15980c9b43e",
                "emojiId": "007"
            }
        ]

        # #text_message = TextSendMessage(text='$ LINE emoji $', emojis=emoji)

        # message = TextSendMessage(text='$ Commands:\n$ info\n$ developers\n$ predict', emojis=emoji)
        message=TextSendMessage(
            text="Pick a following function",
            quick_reply=QuickReply(
                items=[
                    QuickReplyButton(
                        action=MessageAction(label="info",text="info"),
                        image_url="https://cdn1.iconfinder.com/data/icons/info-graphics-2-5/64/one_number_count_track-256.png"
                        ),
                    QuickReplyButton(
                        image_url="https://cdn1.iconfinder.com/data/icons/info-graphics-2-5/64/two_number_count_track-256.png",
                        action=PostbackAction(
                            label="transaction",
                            data = "action=transaction"
                        ),
                        
                        ),
                    QuickReplyButton(
                        action=MessageAction(label="developers",text="developers"),
                        image_url="https://cdn1.iconfinder.com/data/icons/info-graphics-2-5/64/three_number_count_track-256.png"
                        ),
                    ]
                )
            )

        
        line_bot_api.reply_message(event.reply_token, message)


def showDevelopers(userId, event):
    s = "\nBuild Model:李亭臻高為勳Ying-hsuan Chen\nPresentation:Winnie Chang"
    carousel_template_message = TemplateSendMessage(
        alt_text='Carousel template',
        template=CarouselTemplate(
            columns=[
                CarouselColumn(
                    thumbnail_image_url="https://s3.amazonaws.com/content.sitezoogle.com/u/315386/91d4ba9fbceb69ce3b3c3adf42b161ebdcec5915/original/true-professional.jpg",
                    title='All developers',
                    text='Instructors: Ann and Roger\nLineBot Developer:周宥辰',
                    actions=[
                        URITemplateAction(
                            label='高為勳',
                            uri='https://www.facebook.com/wilson.kao.112'
                        ),
                        URITemplateAction(
                            label='李亭臻',
                            uri='https://www.facebook.com/saintpaulstammy'
                        ),
                        URITemplateAction(
                            label='周宥辰',
                            uri='https://www.facebook.com/profile.php?id=100004136469516'
                        ),
                    ]
                ),
                CarouselColumn(
                    thumbnail_image_url='https://s3.amazonaws.com/content.sitezoogle.com/u/315386/91d4ba9fbceb69ce3b3c3adf42b161ebdcec5915/original/true-professional.jpg',
                    title='All developers',
                    text='Build Model:李亭臻,高為勳,陳映璇\nPresentation:Winnie Chang',
                    actions=[
                        URIAction(
                            label='Winnie Chang',
                            uri='https://www.facebook.com/profile.php?id=100000136617830'
                        ),
                        URIAction(
                            label='Ying-hsuan Chen',
                            uri='https://www.facebook.com/profile.php?id=100009206584411'
                        ),
                        PostbackAction(
                            label='contact developer',
                            data="action=contactDeveloper"
                        ),

                    ]
                ),
            ]
        )
    )
    line_bot_api.push_message(userId, carousel_template_message)




def test(user_id):
    profile = line_bot_api.get_profile(user_id)
    line_bot_api.push_message(user_id, TextSendMessage(text=profile.user_id))
    line_bot_api.push_message(user_id, TextSendMessage(text="開始買賣囉\n請問你要買還是賣"))

    print(profile.display_name)
    print(profile.user_id)
    print(profile.picture_url)
    print(profile.status_message)


@handler.add(PostbackEvent)
def handle_postback(event):
    # print("this is handle_postback!!!")
    user_id = event.source.user_id
    data = event.postback.data
    data = data.split("&")
    print(user_id, "trigger postback")
    # print("data in handle_postback", data)
    if data[0]=="action=transaction":
        if len(data)==1:#開始交易後data length > 1
            availableTicket.append(user_id)
        transaction(user_id, data, event)
    elif data[0]=="action=contactDeveloper":
        pass
        #contactDeveloper(event)
        # sent(event)
        # contactDeveloper(event)
def startTimer(t):
    return True


def sent(event):
    print(event)
    user_id = event.source.user_id
    for i in range(0, 10):
        line_bot_api.push_message(developerId, TextSendMessage(text=str(i)))
        time.sleep(1)


def contactDeveloper(event):

    user_id = event.source.user_id
    profile = line_bot_api.get_profile(user_id)
    username = profile.display_name
    picture_url = profile.picture_url
    status_message = profile.status_message
    language = profile.language
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="有人找你\nuserId:"+str(user_id)))
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="有人找你\nuserId:"+str(user_id)))
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="picture_url:"+str(picture_url)))
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="status_message:"+str(status_message)))
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="language:"+str(language)))
    # line_bot_api.push_message(developerId, TextSendMessage(text="有人找你\nuserId:"+str(user_id)))
    # line_bot_api.push_message(developerId, TextSendMessage(text="username:"+str(username)))
    # line_bot_api.push_message(developerId, TextSendMessage(text="picture_url:"+str(picture_url)))
    # line_bot_api.push_message(developerId, TextSendMessage(text="status_message:"+str(status_message)))
    # line_bot_api.push_message(developerId, TextSendMessage(text="language:"+str(language)))



def step1(user_id, data, event):
    #print("This is step1(user_id)")
    ButtonsTemplateM = TemplateSendMessage(
                                alt_text='Buttons template',
                                template=ButtonsTemplate(
                                    title='開始買賣囉',
                                    text='請選擇您想使用的ID',
                                    actions=[
                                        PostbackTemplateAction(
                                            label="ID:1",
                                            data=data[0]+"&1&1"
                                        ),
                                        PostbackTemplateAction(
                                            label='ID:2',
                                            data=data[0]+'&1&2'
                                        ),
                                        PostbackTemplateAction(
                                            label='ID:3',
                                            data=data[0]+'&1&3'
                                        ),
                                        PostbackTemplateAction(
                                            label='ID:4',
                                            data=data[0]+'&1&4'
                                        ),

                                    ]
                                )
    )
    line_bot_api.reply_message(event.reply_token, ButtonsTemplateM)
    # line_bot_api.push_message(user_id, ButtonsTemplateM)

def step2(user_id, data, event):
    
    ButtonsTemplateM = TemplateSendMessage(
                        alt_text='Buttons template',
                        template=ButtonsTemplate(
                            title='您想用ID'+data[2]+"做什麼呢？",
                            text='請選擇您要買或是賣',
                            actions=[
                                PostbackTemplateAction(
                                    label='買',
                                    data=data[0]+"&2&"+data[2]+"&buy"
                                ),
                                PostbackTemplateAction(
                                    label='賣',
                                    data=data[0]+"&2&"+data[2]+"&sell"
                                ),

                            ]
                        )
    )
    line_bot_api.reply_message(event.reply_token, ButtonsTemplateM)
    # line_bot_api.push_message(user_id, ButtonsTemplateM)

def step3(user_id, data, event):
    buyOrSell = "賣"
    if data[3] == "buy":
        buyOrSell = "買"

    ButtonsTemplateM = TemplateSendMessage(
                        alt_text='Buttons template',
                        template=ButtonsTemplate(
                            title='我們提供四檔股票現價交易',
                            text='請選擇您'+buyOrSell+'的股票',
                            actions=[
                                PostbackTemplateAction(
                                    label='股票1',
                                    data= data[0]+"&3&"+data[2]+"&"+data[3]+"&1"
                                ),
                                PostbackTemplateAction(
                                    label='股票2',
                                    data= data[0]+"&3&"+data[2]+"&"+data[3]+"&2"
                                ),
                                PostbackTemplateAction(
                                    label='股票3',
                                    data= data[0]+"&3&"+data[2]+"&"+data[3]+"&3"
                                ),
                                PostbackTemplateAction(
                                    label='股票4',
                                    data= data[0]+"&3&"+data[2]+"&"+data[3]+"&4"
                                ),

                            ]
                        )
    )

    line_bot_api.reply_message(event.reply_token, ButtonsTemplateM)
    # line_bot_api.push_message(user_id, ButtonsTemplateM)

def step4(user_id, data, event):
    ButtonsTemplateM = TemplateSendMessage(
                        alt_text='Buttons template',
                        template=ButtonsTemplate(
                            title="買幾張股票"+data[4]+"呢",
                            text='請選擇張數',
                            actions=[
                                PostbackTemplateAction(
                                    label='1張=1000股',
                                    data= data[0]+"&4&"+data[2]+"&"+data[3]+"&"+data[4]+"&1"
                                ),
                                PostbackTemplateAction(
                                    label='2張=2000股',
                                    data= data[0]+"&4&"+data[2]+"&"+data[3]+"&"+data[4]+"&2"
                                ),
                                PostbackTemplateAction(
                                    label='3張=3000股',
                                    data= data[0]+"&4&"+data[2]+"&"+data[3]+"&"+data[4]+"&3"
                                ),
                                PostbackTemplateAction(
                                    label='4張=4000股',
                                    data= data[0]+"&4&"+data[2]+"&"+data[3]+"&"+data[4]+"&4"
                                ),

                            ]
                        )
    )

    line_bot_api.reply_message(event.reply_token, ButtonsTemplateM)
    # line_bot_api.push_message(user_id, ButtonsTemplateM)

def judge(user_id, data, event):
    id = data[2]
    stockNo = data[4]
    shares = float(data[5])*1000.0
    if data[3] == "buy":
        buyOrSell = 1
    print("result of predict: ", open.startPredict(id, buyOrSell, stockNo, shares))
    risk = "低"
    buyOrSell = "賣"
    alert = "\n您的違約風險是"+risk+"的"
    username = line_bot_api.get_profile(user_id).display_name

    if data[3] == "buy":
        buyOrSell = "買"

    if data[2]=="1" and data[4]=="1": #userID4 buy stock1
        risk = "高"
        alert = "\n您的違約風險是"+risk+"的"
    if data[3] == "buy":#買股票才會違約
        confirm_template_message = TemplateSendMessage(
            alt_text='Confirm template',
            template=ConfirmTemplate(
                text=username+" 想使用ID"+data[2]+" "+buyOrSell+"了股票"+data[4]+"共"+data[5]+"張"+alert+"\n確定購買？",
                actions=[
                    PostbackAction(
                        label='是，我錢夠',
                        data=data[0]+'&finish&'+data[2]+"&"+data[3]+"&"+data[4]+"&"+data[5]
                    ),
                    PostbackAction(
                        label='否，湊不出錢',
                        data=data[0]+'&unfinish'
                    ),
                ]
            )
        )
        # line_bot_api.push_message(user_id, confirm_template_message)
        line_bot_api.reply_message(event.reply_token, confirm_template_message)

        

    else: #賣股票不會違約

        text_message = TextSendMessage(text="感謝 "+username+" 使用ID"+data[2]+" "+buyOrSell+"了股票"+data[4]+"共"+data[5]+"張")
        # line_bot_api.push_message(user_id, text_message)
        line_bot_api.reply_message(event.reply_token, text_message)
        quit(user_id)
        
def transaction(user_id, data, event):
    # print("this is transaction()!!!")
    print("data in transaction():", data)
    # print("availableTicket in transaction():", availableTicket)
    
    username = line_bot_api.get_profile(user_id).display_name
    if not(user_id in availableTicket):
        # line_bot_api.push_message(user_id, TextSendMessage(text="此單已結束\n請用USER GUIDE重新交易"))
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="此單已結束\n請用USER GUIDE重新交易"))
        return

    elif len(data)==1:
        step1(user_id, data, event) #選虛擬使用者
        #print("data after step1():", data)
    if len(data)>1:
        if data[1] == "1": #選好虛擬使用者，要選買or賣
            step2(user_id, data, event)
            #print("data after step2():", data)

        elif data[1] == "2": #完成第二步，已知買or賣，要挑股票了
            step3(user_id, data, event)
            #print("data after step3():", data)
        elif data[1] == "3":
            step4(user_id, data, event)
        elif data[1] == "4": #完成第三步，已知交易哪支股票，要判斷了
            print("start judge!")
            judge(user_id, data, event)
            #print("data after judge():", data)
        elif data[1]=='finish':
            buyOrSell = "賣"
            if data[3] == "buy":
                buyOrSell = "買"
            username = line_bot_api.get_profile(user_id).display_name
            text_message = TextSendMessage(text="感謝 "+username+" 使用ID"+data[2]+" "+buyOrSell+"了股票"+data[4]+"共"+data[5]+"張")
            # line_bot_api.push_message(user_id, text_message)
            line_bot_api.reply_message(event.reply_token, text_message)
            quit(user_id)
        elif data[1]=='unfinish':
            text_message = TextSendMessage(text='已經放棄此次交易')
            # line_bot_api.push_message(user_id, text_message)
            line_bot_api.reply_message(event.reply_token, text_message)
            quit(user_id)

def quit(user_id):
    print(availableTicket)
    print("quit:"+user_id)
    for i in range(0, len(availableTicket)+1):
        if user_id in availableTicket:
            availableTicket.remove(user_id)
    print(availableTicket)
    return













def handle_postback2(event):
    print("this is handle_postback!!!")
    user_id = event.source.user_id
    username = line_bot_api.get_profile(user_id).display_name
    
    data = event.postback.data
    data = data.split("&")
    print("data:", data)
    print("availableTicket:", availableTicket)

    if user_id in availableTicket:
        line_bot_api.push_message(user_id, TextSendMessage(text="user_id in availableTicket"))
    if data[0] == "quit":
        line_bot_api.push_message(user_id, TextSendMessage(text=username+ "您已放棄此次操作"))

    if data[0] == "1": #選好虛擬使用者
        ButtonsTemplateM = TemplateSendMessage(
                            alt_text='Buttons template',
                            template=ButtonsTemplate(
                                title='您想用ID'+data[1]+"做什麼呢？",
                                text='請選擇您要買或是賣',
                                actions=[
                                    PostbackTemplateAction(
                                        label='買',
                                        data="2&"+data[1]+"&buy"
                                    ),
                                    PostbackTemplateAction(
                                        label='賣',
                                        data="2&"+data[1]+"&sell"
                                    ),

                                ]
                            )
        )
        line_bot_api.reply_message(event.reply_token, ButtonsTemplateM)


    if data[0] == "2":#完成第二步，已知買or賣
        buyOrSell = "賣"
        if data[2] == "buy":
            buyOrSell = "買"

        ButtonsTemplateM = TemplateSendMessage(
                            alt_text='Buttons template',
                            template=ButtonsTemplate(
                                title='我們提供四檔股票現價交易',
                                text='請選擇您'+buyOrSell+'的股票',
                                actions=[
                                    PostbackTemplateAction(
                                        label='股票1',
                                        data= "3&"+data[1]+"&"+data[2]+"&1"
                                    ),
                                    PostbackTemplateAction(
                                        label='股票2',
                                        data= "3&"+data[1]+"&"+data[2]+"&2"
                                    ),
                                    PostbackTemplateAction(
                                        label='股票3',
                                        data= "3&"+data[1]+"&"+data[2]+"&3"
                                    ),
                                    PostbackTemplateAction(
                                        label='股票4',
                                        data= "3&"+data[1]+"&"+data[2]+"&4"
                                    ),

                                ]
                            )
        )

        line_bot_api.reply_message(event.reply_token, ButtonsTemplateM)


    if data[0] == "3":
        buyOrSell = "賣"
        alert = "\n您的違約風險是低的"

        if data[2] == "buy":
            buyOrSell = "買"

        if data[1]=="1" and data[3]=="1":
            alert = "\n您的違約風險是高的"
            if data[2] == "buy":#買股票才會違約
                confirm_template_message = TemplateSendMessage(
                    alt_text='Confirm template',
                    template=ConfirmTemplate(
                        text="感謝 "+username+" 使用ID"+data[1]+" "+buyOrSell+"了股票"+data[3]+alert+"\n確定購買？",
                        actions=[
                            PostbackAction(
                                label='是，我錢夠',
                                data='finish'
                            ),
                            PostbackAction(
                                label='否，湊不出錢',
                                data='unfinish'
                            ),
                        ]
                    )
                )
            line_bot_api.reply_message(event.reply_token, confirm_template_message)

        else: #賣股票不會違約
            text_message = TextSendMessage(text='已經完成此次交易')
            line_bot_api.push_message(user_id, text_message)











if __name__ == "__main__":
    availableTicket = []
    # for i in range(100, 140):
    #     availableTicket.append(str(i))
    # print("availableTicket in main:", availableTicket)
    developerId = 'U05d832026e5dd3f2c32ae70b8500c400'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)