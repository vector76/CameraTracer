// viewmodel copied from sample at https://docs.octoprint.org/en/master/plugins/gettingstarted.html#id11
// but with name changed to TracerViewModel

$(function() {
    function TracerViewModel(parameters) {
        var self = this;
        
        self.imagecount = 0;

        self.settings = parameters[0];

        self.camSrc = ko.observable();
        self.camSrc("/plugin/tracer/camera_image");
        self.skelSrc = ko.observable();
        self.skelSrc("/plugin/tracer/camera_aim?");

        self.takePicture = function() {
            // alert("called takePicture");
            self.imagecount = self.imagecount + 1;
            self.camSrc("/plugin/tracer/camera_image?" + self.imagecount);
            self.skelSrc("/plugin/tracer/camera_aim?" + self.imagecount);
        }

        // This will get called before the HelloWorldViewModel gets bound to the DOM, but after its
        // dependencies have already been initialized. It is especially guaranteed that this method
        // gets called _after_ the settings have been retrieved from the OctoPrint backend and thus
        // the SettingsViewModel been properly populated.
        self.onBeforeBinding = function() {
            console.log("called onBeforeBinding");
        }
    }

    // This is how our plugin registers itself with the application, by adding some configuration
    // information to the global variable OCTOPRINT_VIEWMODELS
    OCTOPRINT_VIEWMODELS.push([
        // This is the constructor to call for instantiating the plugin
        TracerViewModel,

        // This is a list of dependencies to inject into the plugin, the order which you request
        // here is the order in which the dependencies will be injected into your view model upon
        // instantiation via the parameters argument
        ["settingsViewModel"],

        // Finally, this is the list of selectors for all elements we want this view model to be bound to.
        ["#tab_plugin_tracer"]
    ]);
});