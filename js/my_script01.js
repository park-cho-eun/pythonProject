function say_hello() {
    var name = document.getElementById("name").value;
    var message = "<h2>Hello "+ name +"!</h2>";

    document.getElementById("content").innerHTML = message;

    if (name=='student'){
        var title = document.getElementById("title").innerHTML ;
        /* var title = document.querySelector(selectors: "#title").innerHTML ;*/
        title += " & Loving it!" ;
        document.getElementById("title").innerHTML = title;
    }
}