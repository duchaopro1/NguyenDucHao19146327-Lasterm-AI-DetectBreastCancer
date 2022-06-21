
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("Hue.h5") #model m train

selected = option_menu(None, ["Diagnostic", "More"], 
    icons=[ "upload", 'bookmark-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"}
    }
)
if selected == "Diagnostic":
            ### load file
   uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])

   map_dict = {0: 'BENIGN',
            1: 'MALIGNANT',
            2: 'NORMAL'}
    
 
   if uploaded_file is not None:
    # Convert the file
            img = image.load_img(uploaded_file,target_size=(64,64)) #xử lí ảnh theo cách m làm
            ima = image.load_img(uploaded_file,target_size=(250,250))
            st.image(ima, channels="RGB") #hiển thị ảnh
            img = img_to_array(img)
            img = img.reshape(1,64,64,3)
            img = img.astype('float32')
            img = img/255
                
    #Button: nút dự đoán sau khi up ảnh
            Genrate_pred = st.button("Generate Prediction") 
    
            if Genrate_pred:    
                  prediction = model.predict(img).argmax()
                  st.write("**Predicted Label for the image is {}**".format(map_dict[prediction]))
                  if(prediction == 1): 
                    st.write("Bạn có một khối u không mấy hiền lành :< Nhưng mà đừng lo vì chúng ta đã phát hiện ra nó <3 Hãy để bác sĩ tư vấn cho bạn nhé <3 It will be okay")
   
if selected == "More":
    st.title("BỆNH UNG THƯ VÚ")
    st.title("------------------------------------------------")
    st.write("BVK - Theo thống kê của GLOBOCAN năm 2020, nước ta có gần 183.000 số ca mắc mới ung thư, trong đó ung thư vú là 21.555 người mắc chiếm tỷ lệ 11,8%. Ngoài ra cũng trong năm 2020, Việt Nam ghi nhận hơn 9.345 trường hợp tử vong vì căn bệnh này. Đây thực sự là con số đáng lo ngại, bởi ung thư vú cũng là bệnh ung thư thường gặp nhất ở nữ giới, là mối quan tâm của nhiều chị em phụ nữ.")
    st.header("Viêm phổi là gì?")
    st.write("**Viêm phổi** là tình trạng viêm nhiễm (sưng) nhu mô phổi bao gồm viêm phế nang (túi khí nhỏ), túi phế nang, ống phế nang, tổ chức liên kết khe kẽ và viêm tiểu phế quản tận cùng, chủ yếu do vi khuẩn, virus, nấm gây nên. Các phế nang, đường dẫn khí chứa nhiều dịch nhầy hoặc mủ, xuất tiết dịch đường hô hấp trên gây ho đờm, sốt ớn lạnh, khó thở. Hiện tượng viêm phổi có thể ở một vùng hoặc ở một vài vùng (viêm phổi thùy hoặc “đa thùy”) hoặc toàn bộ phổi.")
    st.header("Nguyên nhân gây bệnh viêm phổi")
    st.subheader("1. Viêm phổi mắc phải cộng đồng")
    st.write("""**+ Viêm phổi do vi khuẩn**": "**Vi khuẩn** là nguyên nhân thường gặp nhất gây viêm phổi ở trẻ em và người lớn. Viêm phổi do vi khuẩn nếu không nhận biết sớm và điều trị kịp thời sẽ dễ dẫn đến hậu quả khó lường, thậm chí tử vong. Các loại vi khuẩn thường gặp gồm: Streptococcus pneumoniae, Legionella pneumophila, Haemophilus influenzae, Mycoplasma pneumoniae, Chlamydia pneumoniae,…""")
    st.write("""**+ Viêm phổi do virus (bao gồm Covid-19)**: Hiện nay, virus SARS-CoV-2 là tác nhân nguy hiểm nhất gây viêm phổi, trong đó, virus có thể làm hỏng phế nang và khiến chất lỏng tích tụ trong phổi. Điều đó cũng có thể dẫn đến sự phát triển của hội chứng suy hô hấp cấp tính (ARDS) – một dạng suy hô hấp nghiêm trọng.""")
    st.write("""**+ Viêm phổi do nấm**:"Viêm phổi do nấm là tình trạng bệnh nhân hít phải bào tử của nấm gây viêm nhiễm, ảnh hưởng nghiêm trọng đến hệ hô hấp với mức độ phát triển nhanh. Bệnh thường có diễn biến phức tạp nếu không được điều trị kịp thời có thể gây ra những biến chứng nguy hiểm thậm chí thiệt mạng.""")
    st.write("""**+ Viêm phổi do hóa chất**:
             Viêm phổi do hóa chất thường hiếm gặp, ít xảy ra nhưng cực kỳ nguy hiểm vì tỷ lệ gây tử vong cao cho người bệnh. Tùy thuộc vào loại hóa chất đã phơi nhiễm mà có mức độ nguy hiểm khác nhau. Bên cạnh tổn thương phổi, các hóa chất còn có thể gây hại cho nhiều cơ quan khác. Chính vì vậy việc phòng ngừa các tác nhân gây bệnh viêm phổi là việc làm cấp thiết, tránh những hậu quả đáng tiếc xảy ra.""")
    st.subheader("2. Viêm phổi mắc phải ở bệnh viện")
    st.write("Theo nghiên cứu, viêm phổi bệnh viện ở các nước phát triển chiếm tỷ lệ 15% trong tổng số trường hợp nhiễm khuẩn bệnh viện và chiếm 27% xảy ra nhiễm khuẩn này ở khoa hồi sức cấp cứu. Những vi khuẩn gây ra tình trạng này là P. aeruginosa, Acinetobacter spp, Enterobacteriacae, Haemophillus spp, S. aureus (MRSA), Streptococcus spp,…")
    st.write("Tại Việt Nam, viêm phổi bệnh viện chiếm tỷ lệ khoảng từ 21% đến 75%, trong đó viêm phổi do lây nhiễm qua thở máy chiếm đến 90% và được xác định sau thở máy 48 giờ. Đây là một vấn đề rất khó khăn mà các khoa lâm sàng, đặc biệt là khoa hồi sức tích cực phải đương đầu vì khó chẩn đoán, kéo dài thời gian điều trị, tốn kém rất nhiều chi phí.")
    st.subheader("3. Viêm phổi do hít thở")
    st.write("**Nguyên nhân gây viêm phổi** ở trường hợp này là do người bệnh hít phải lượng lớn dị vật từ đường thở (miệng, hầu họng, dạ dày,…) và dị vật rơi vào phổi 2 bên. Các dị vật hít phải có thể là nước bọt, thức ăn, hóa chất, axit dịch vị,… nếu chúng đi vào phổi sẽ kích thích phản ứng viêm của niêm mạc phổi, tạo cơ hội để vi khuẩn xâm nhập và gây viêm phổi.")
    st.header("Thời gian ủ bệnh viêm phổi bao lâu?")
    st.write("Phần lớn trường hợp, viêm phổi thường xuất hiện ở dạng cấp tính (bệnh kéo dài dưới 6 tuần) với các triệu chứng rất rõ ràng ở những ngày đầu mới phát bệnh. Đặc biệt, nếu tình trạng khó thở càng trở nặng thì nguy cơ tử vong trong thời gian ngắn càng cao.")
    st.write("Mặt khác, viêm phổi mạn tính cũng có biểu hiện tương tự, thời gian bệnh kéo dài không dứt. Một người được chẩn đoán mắc viêm phổi mạn tính khi bệnh kéo dài quá 6 tuần.")
    st.header("Dấu hiệu viêm phổi thường gặp")
    st.write("Các triệu chứng viêm phổi thường gặp như: Đau ngực khi thở hoặc ho; Ho, ho khan, ho có đờm; Sốt trên 38 độ, đổ mồ hôi và ớn lạnh; Mệt mỏi, uể oải và chán ăn; Thở nhanh, khó thở khi gắng sức; Buồn nôn, nôn mửa hoặc tiêu chảy;")
    st.title("Để biết thêm thông tin chi tiết: https://vnvc.vn/benh-viem-phoi-nguyen-nhan-trieu-chung-dieu-tri-va-cach-phong-ngua/")
    
