
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
    st.write("**BVK - Theo thống kê của GLOBOCAN năm 2020, nước ta có gần 183.000 số ca mắc mới ung thư, trong đó ung thư vú là 21.555 người mắc chiếm tỷ lệ 11,8%. Ngoài ra cũng trong năm 2020, Việt Nam ghi nhận hơn 9.345 trường hợp tử vong vì căn bệnh này. Đây thực sự là con số đáng lo ngại, bởi ung thư vú cũng là bệnh ung thư thường gặp nhất ở nữ giới, là mối quan tâm của nhiều chị em phụ nữ.**")
    st.write("Nếu như trước đây, người mắc ung thư vú thường gặp ở độ tuổi trên 40 thì giờ đây căn bệnh này có xu hướng trẻ hóa. Hãy cùng tìm hiểu rõ hơn về căn bệnh này qua bài viết dưới đây.")
    st.header("Ung thư vú là gì?")
    st.write("**Ung thư vú** xảy ra khi các tế bào ung thư ác tính được phát hiện trong mô vú, tình trạng này thường phát triển từ các ống bên trong vú. Các tế bào ung thư sau đó có thể lan rộng ra toàn bộ vú và sang các bộ phận khác trên cơ thể.")
    st.subheader("1. Dấu hiệu ban đầu của bênh")
    st.write("Ung thư vú có thể bao gồm các triệu chứng sau:")
    st.write("- Cục u không đau ở vú")
    st.write("- Ngứa và phát ban kéo dài quanh núm vú")
    st.write("- Chảy máu hoặc tiết dịch bất thường ở núm vú")
    st.write("- Vùng da trên vú sưng và dày lên")
    st.write("- Vùng da trên vú sần vỏ cam hoặc nhăn nheo")
    st.write("- Núm vú bị tụt hoặc lõm vào trong")
    st.write("- Vùng nách sưng, đau hoặc có u")
    st.subheader("2. Yếu tố nguy cơ gây ung thư vú")
    st.write("""**- Độ tuổi**": Ung thư vú có thể gặp ở mọi lứa tuổi, đặc biệt là những phụ nữ trên 45 tuổi. Đặc biệt, những phụ nữ không sinh con và sinh con đầu lòng sau độ tuổi 30 có nguy cơ mắc ung thư vú cao hơn những người bình thường.""")
    st.write("""**- Bản thân mắc bệnh lý về tuyến vú**: như xơ vú, áp – xe – vú… nếu không được điều trị kịp thời sẽ dẫn đến những tổn thương khó hồi phục ở vùng vú và tiến triển thành ung thư.""")
    st.write("""**- Yếu tố di truyền**: trong gia đình nếu có bà, mẹ hay chị gái mắc ung thư vú thì tỷ lệ mắc ung thư vú của cá nhân đó sẽ cao hơn. Phần lớn các trường hợp ung thư vú do di truyền thường từ 2 gen BRCA1 và BRCA2. Những phụ nữ có đột biến gen BRCA1 và/hoặc BRCA2 có thể có đến 80% nguy cơ mắc bệnh.""")
    st.write("""**- Người từng bị ung thư**: như ung thư buồng trứng, phúc mạc, vòi trứng hoặc đã từng xạ trị vùng ngực cũng có nguy cơ bị ung thư vú cao.""")
    st.write("""**- Phụ nữ dậy thì sớm**: (trước 12 tuổi) và mãn kinh muộn (sau 55 tuổi) cũng có nguy cơ mắc bệnh ung thư vú cao hơn người khác. Nguyên nhân là do những phụ nữ này chịu tác động lâu dài của hormone estrogen và progesterone.""")
    st.write("""**- Béo phì**: cũng là một yếu tố làm tăng nguy cơ mắc bệnh ung thư vú. Nguyên nhân là do phụ nữ bị béo phì thường sản sinh ra nhiều estrogen hơn so với phụ nữ khác. Béo phì không chỉ làm tăng nguy cơ dẫn đến ung thư vú mà còn làm gia tăng nguy cơ mắc các bệnh tim mạch, mỡ máu và các bệnh ung thư khác như ung thư buồng trứng, ung thư đại trực tràng, ung thư gan,…""")
    st.write("""**- Lối sống và sinh hoạt thiếu khoa học**: Chế độ ăn uống nhiều calo trong khi cơ thể lười vận động sẽ làm lượng mỡ thừa trong cơ thể tăng cao dẫn đến béo phì và làm tăng nguy cơ mắc ung thư vú. Ngoài ra, hút thuốc lá, uống nhiều rượu bia, căng thẳng kéo dài cũng dễ dẫn đến ung thư vú.""")
    st.write("""**- Phơi nhiễm phóng xạ**: Tuy lượng phơi nhiễm từ tia X là rất thấp nhưng nữ giới cũng cần hạn chế tiếp xúc với môi trường phóng xạ để tránh nguy cơ mắc bệnh.""")
    st.subheader("3. Điều trị ung thư vú")
    st.write("Ngày nay, với tiến bộ của y học hiện đại đã mang đến nhiều giải pháp điều trị hiệu quả hơn cho bệnh ung thư vú.")
    st.write("Điều trị ung thư vú dựa trên nguyên tắc điều trị là đa mô thức tức là kết hợp các phương pháp điều trị phẫu thuật, hóa trị, xạ trị, điều trị nội tiết .... tùy vào giai đoạn bệnh, độ tuổi, tình trạng sức khỏe và nguyện vọng của người bệnh.")
    st.write("""**Phẫu thuật**: là cắt bỏ khối u toàn bộ hoặc một phần, và có thể loại bỏ hạch nách khi cần.""")
    st.write("""**Xạ trị**: là chiếu tia bức xạ vào vùng bệnh nhằm mục tiêu phá hủy tế bào ung thư. Xạ trị có thể được thực hiện sau phẫu thuật hoặc sau hóa trị.""")
    st.write("""**Liệu pháp nội tiết**: là điều trị quan trọng nhất dành cho trường hợp UTV có thụ thể nội tiết ER (+) và/hoặc PR (+). Thuốc ức chế hoặc ngăn chận tác động của các hormon nội tiết – được biết là có liên quan đến quá trình tăng sinh tế bào ung thư. Liệu pháp nội tiết có thể được sử dụng sau phẫu thuật hoặc giai đoạn muộn.""")
    st.write("""**Liệu pháp kháng HER2**: à điều trị quan trọng dành cho trường hợp UTV có HER2 dương tính. Thuốc ức chế tác động của các thụ thể HER2 – được biết là có liên quan quá trình tăng sinh tế bào ung thư.""")
    st.title("Để biết thêm thông tin chi tiết: https://benhvienk.vn/ung-thu-vu-va-nhung-dieu-ban-khong-nen-bo-qua-nd91360.html/")
    
