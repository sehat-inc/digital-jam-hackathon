document.addEventListener('DOMContentLoaded', function() {
//     // Cursor following animation
//     const animatedObject = document.createElement('div');
//     animatedObject.id = 'animated-object';
//     document.body.appendChild(animatedObject);

//     document.addEventListener('mousemove', function(e) {
//         const mouseX = e.clientX;
//         const mouseY = e.clientY;
//          animatedObject.style.transform = `translate(${mouseX-10}px, ${mouseY-10}px)`;
//     });


 //Chat input and Output
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send');
    const chatOutput = document.getElementById('chat-output');
    if(chatSend && chatInput && chatOutput){
            chatSend.addEventListener('click', function() {
                const message = chatInput.value;
                if (message.trim() !== '') {
                    chatOutput.innerHTML += `<p>You: ${message}</p>`;
                    // Here, you would integrate your AI chatbot logic
                   setTimeout(function() {
                        chatOutput.innerHTML += `<p>AI: Response is not implemented yet. <p>`;
                         chatOutput.scrollTop = chatOutput.scrollHeight;
                   }, 1000);
                   chatOutput.scrollTop = chatOutput.scrollHeight;
                    chatInput.value = '';
                }
            });
    }
});