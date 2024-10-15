package tools.utils.kappalab;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.Socket;
import java.util.concurrent.Callable;

import com.google.gson.Gson;

/**
 * Communicates with the Kappalab server through a socket.
 * 
 * @param input The input parameters for the Kappalab computation.
 */
public class KappalabSocket implements Callable<KappalabOutput> {

    /**
     * The default port for the Kappalab socket.
     */
    public static final int KAPPALAB_SOCKET_PORT = 6011;

    private final KappalabInput input;
    private final Gson gson = new Gson();

    public KappalabSocket(KappalabInput input) {
        this.input = input;
    }

    /**
     * Calls the Kappalab server through a socket connection and retrieves the
     * output.
     *
     * @return The KappalabOutput containing the computed results.
     * @throws Exception If an error occurs during the socket communication.
     */
    @Override
    public KappalabOutput call() throws Exception {
        try (Socket socket = new Socket("localhost", KAPPALAB_SOCKET_PORT);
                BufferedWriter os = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
                BufferedReader is = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {

            // Send KappalabInput to the server
            os.write(gson.toJson(input));
            os.newLine();
            os.flush();

            // Receive the server's response
            String response = is.readLine();

            // Parse the response into KappalabOutput
            KappalabOutput output = gson.fromJson(response, KappalabOutput.class);

            return output;
        }
    }
}