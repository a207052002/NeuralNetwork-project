改用 py 64 bits 開啟時間略久

完善了 cross-entropy-softmax 的 backpropagation 目前所有資料都推薦使用 CN+SM，並且如果選了 CN 做 Loss 這就是是強制的

使用 Xaiver 優化權重初始以適應 relu
雖然是 relu，但其實裡面是 ELU，alpha 是 0.5，因為實測後效果比較好，雖然比較容易容易因為資料稍微極端導致一時 overflow/underflow

填入數值的方式都可以比照剛打開程式的時候給的範例去填，新增了 mini-batch 的功能，只要不是使用 SGD 都會自動套用
所以如果不使用 SGD 又沒有給 batch_size 程式會出錯，這時請重新讀取一個資料刷新整個面板

除了 4satellite-6 需要較久的時間 training 外，所有資料在 Softmax + Cross entropy 的效果下需要的 training 次數非常之少
所以都能夠手動進行測試不需放置，除非深度跟寬度條過於極端，或是 lr 調得不好
大抵上使用 adam 都能很快達到該資料的最高期望準確度(有些資料註定無法 100%)
